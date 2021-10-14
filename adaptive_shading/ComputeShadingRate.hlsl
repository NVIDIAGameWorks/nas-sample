#pragma pack_matrix(row_major)

#include "Compute_cb.h"

cbuffer ShadingRatePassCB : register(b0)
{
    AdaptiveShadingConstants ShadingRatePassParams;
};


RWTexture2D<uint> vrsSurface : register(u0);
Texture2D<float> gBufferDepth : register(t0);
Texture2D<float2> nasDataSurface : register(t1);
SamplerState s_Sampler : register(s0);

groupshared uint groupMinDepth;

#define TILE_SIZE 16

[numthreads(8, 4, 1)]
void main_cs(uint3 DispatchThreadID : SV_DispatchThreadID, uint3 GroupThreadID : SV_GroupThreadID, uint3 GroupID : SV_GroupID)
{
    uint screenWidth, screenHeight;
    gBufferDepth.GetDimensions(screenWidth, screenHeight);

    if (all(GroupThreadID.xy == 0))
    {
        groupMinDepth = asuint(1.0f);
    }
    GroupMemoryBarrierWithGroupSync();

    // Block of 8x4 threads (each thread is a block of 2x4 pixels)
    // Each block is responsible for loading data from a 16x16 pixel tile
    uint2 localID = GroupThreadID.xy;
    localID <<= uint2(1, 2);

    // Tile global location
    uint2 tileOffset = GroupID.xy << 4;

    // Global block coordinates
    int3 blockBaseCoord = int3(tileOffset + localID, 0);

    // Sample depth in the 2x4 block of each thread
    // sparsely sampling only four of the eight samples
    float4 depth;
    depth.x = gBufferDepth.Load(blockBaseCoord, int2(0, 0)).x;
    depth.y = gBufferDepth.Load(blockBaseCoord, int2(1, 1)).x;
    depth.z = gBufferDepth.Load(blockBaseCoord, int2(0, 2)).x;
    depth.w = gBufferDepth.Load(blockBaseCoord, int2(1, 3)).x;

    // Reduction: find block minimum depth (corresponding to largest motion in block)
    depth.xy = min(depth.xy, depth.zw);
    depth.x = min(depth.x, depth.y);
    InterlockedMin(groupMinDepth, asuint(depth.x));

    GroupMemoryBarrierWithGroupSync();

    if (all(GroupThreadID.xy == 0))
    {
        // Compute motion vector by reconstructing and reprojecting clipPos of the tile center
        // currWindowPos assumes only a single view, safe for non-stereo cases
        float2 currWindowPos = (GroupID.xy + 0.5) * TILE_SIZE;
        float2 currUv = currWindowPos * ShadingRatePassParams.sourceTextureSizeInv;

        float4 clipPos;
        clipPos.x = currUv.x * 2 - 1;
        clipPos.y = 1 - currUv.y * 2;
        clipPos.z = asfloat(groupMinDepth);
        clipPos.w = 1;

        float2 mVec = float2(0, 0);
        float4 prevClipPos = mul(clipPos, ShadingRatePassParams.reprojectionMatrix);

        float2 prevWindowPos = currWindowPos;

        if (prevClipPos.w > 0)
        {
            prevClipPos.xyz /= prevClipPos.w;
            float2 prevUV;
            prevUV.x = 0.5 + prevClipPos.x * 0.5;
            prevUV.y = 0.5 - prevClipPos.y * 0.5;

            prevWindowPos = prevUV * ShadingRatePassParams.previousViewSize + ShadingRatePassParams.previousViewOrigin;
            mVec = prevWindowPos.xy - currWindowPos.xy;
        }

        mVec = abs(mVec) * ShadingRatePassParams.motionSensitivity;

        // Error scalers (equations from the I3D 2019 paper)
        // bhv for half rate, bqv for quarter rate
        float2 bhv = pow(1.0 / (1 + pow(1.05 * mVec, 3.1)), 0.35);
        float2 bqv = 2.13 * pow(1.0 / (1 + pow(0.55 * mVec, 2.41)), 0.49);

        // Sample block error data from NAS data pass and apply the error scalars
        float2 diff = nasDataSurface.SampleLevel(s_Sampler, prevWindowPos * ShadingRatePassParams.sourceTextureSizeInv, 0).rg;
        float2 diff2 = diff * bhv;
        float2 diff4 = diff * bqv;

        float threshold = ShadingRatePassParams.errorSensitivity;

        /*
            D3D12_SHADING_RATE_1X1	= 0,   // 0b0000
            D3D12_SHADING_RATE_1X2	= 0x1, // 0b0001
            D3D12_SHADING_RATE_2X1	= 0x4, // 0b0100
            D3D12_SHADING_RATE_2X2	= 0x5, // 0b0101
            D3D12_SHADING_RATE_2X4	= 0x6, // 0b0110
            D3D12_SHADING_RATE_4X2	= 0x9, // 0b1001
            D3D12_SHADING_RATE_4X4	= 0xa  // 0b1010
        */

        // Compute block shading rate based on if the error computation goes over the threshold
        // shading rates in D3D are purposely designed to be able to combined, e.g. 2x1 | 1x2 = 2x2
        uint ShadingRate = 0;
        ShadingRate |= ((diff2.x >= threshold) ? 0 : ((diff4.x > threshold) ? 0x4 : 0x8));
        ShadingRate |= ((diff2.y >= threshold) ? 0 : ((diff4.y > threshold) ? 0x1 : 0x2));

        // Disable 4x4 shading rate (low quality, limited perf gain)
        if (ShadingRate == 0xa)
        {
            ShadingRate = (diff2.x > diff2.y) ? 0x6 : 0x9; // use 2x4 or 4x2 based on directional gradient
        }
        // Disable 4x1 or 1x4 shading rate (unsupported)
        else if (ShadingRate == 0x8)
        {
            ShadingRate = 0x4;
        }
        else if (ShadingRate == 0x2)
        {
            ShadingRate = 0x1;
        }

        vrsSurface[GroupID.xy] = ShadingRate;
    }
}

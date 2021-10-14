RWTexture2D<float2> nasDataSurface : register(u0);
Texture2D<float4> prevFrameColors : register(t0);

#include "Compute_cb.h"

cbuffer ComputeNasDataPassCB : register(b0)
{
    ComputeNASDataConstants ComputeNASDataParams;
};

float RgbToLuminance(float3 color)
{
    return dot(color, float3(0.299, 0.587, 0.114));
}

// Use a single wave threadgroup to leverage wave intrinsics
[numthreads(8, 4, 1)]
void main_cs(uint3 DispatchThreadID : SV_DispatchThreadID, uint3 GroupThreadID : SV_GroupThreadID, uint3 GroupID : SV_GroupID)
{
    // Block of 8x4 threads (each thread is a block of 2x4 pixels)
    // Each block is responsible for loading data from a 16x16 pixel tile
    uint2 localID = GroupThreadID.xy;
    localID <<= uint2(1, 2);

    // Tile global location
    uint2 tileOffset = GroupID.xy << 4;

    // Global block coordinates
    int3 blockBaseCoord = int3(tileOffset + localID, 0);

    // Fetch color (final post-AA) data
    // l0.x  l0.y
    // l0.z  l0.w  l2.x
    // l1.x  l1.y
    // l1.z  l1.w  l2.y
    //		 l2.z
    float4 l0;
    l0.x = RgbToLuminance(prevFrameColors.Load(blockBaseCoord, int2(0, 0)).xyz);
    l0.y = RgbToLuminance(prevFrameColors.Load(blockBaseCoord, int2(1, 0)).xyz);
    l0.z = RgbToLuminance(prevFrameColors.Load(blockBaseCoord, int2(0, 1)).xyz);
    l0.w = RgbToLuminance(prevFrameColors.Load(blockBaseCoord, int2(1, 1)).xyz);

    float4 l1;
    l1.x = RgbToLuminance(prevFrameColors.Load(blockBaseCoord, int2(0, 2)).xyz);
    l1.y = RgbToLuminance(prevFrameColors.Load(blockBaseCoord, int2(1, 2)).xyz);
    l1.z = RgbToLuminance(prevFrameColors.Load(blockBaseCoord, int2(0, 3)).xyz);
    l1.w = RgbToLuminance(prevFrameColors.Load(blockBaseCoord, int2(1, 3)).xyz);

    float3 l2;
    l2.x = RgbToLuminance(prevFrameColors.Load(blockBaseCoord, int2(2, 1)).xyz);
    l2.y = RgbToLuminance(prevFrameColors.Load(blockBaseCoord, int2(2, 3)).xyz);
    l2.z = RgbToLuminance(prevFrameColors.Load(blockBaseCoord, int2(1, 4)).xyz);

    // Derivatives X
    float4 a = float4(l0.y, l2.x, l1.y, l2.y);
    float4 b = float4(l0.x, l0.w, l1.x, l1.w);
    float4 dx = abs(a - b);

    // Derivatives Y
    a = float4(l0.z, l1.y, l1.z, l2.z);
    b = float4(l0.x, l0.w, l1.x, l1.w);
    float4 dy = abs(a - b);

    // Compute block average luma (8 total samples)
    float4 sumAB = l0 + l1;
    float avgLuma = (sumAB.x + sumAB.y + sumAB.z + sumAB.w) / 8;
    avgLuma = WaveActiveSum(avgLuma) / WaveGetLaneCount() + ComputeNASDataParams.brightnessSensitivity;

    // Compute maximum partial derivative of all 16x16 pixels (256 total)
    // one thread works on 2x4 pixels, one wave has 32 threads, 2x4x32 = 256
    // this approach is more "sensitive" to individual outliers in a tile, since it takes the max instead of the average
    float maxDx = max(max(dx.x, dx.y), max(dx.z, dx.w));
    float maxDy = max(max(dy.x, dy.y), max(dy.z, dy.w));
    float errX = WaveActiveMax(maxDx);
    float errY = WaveActiveMax(maxDy);

    /*
    // Alternative: compute block error using L2 norm; this is the original approach from the paper
    // (Note: errorSensitivity threshold needs to be reduced to get similar quality)
    float sumDxSq = dot(dx * dx, float4(1.0, 1.0, 1.0, 1.0)) / 4.0;
    float sumDySq = dot(dy * dy, float4(1.0, 1.0, 1.0, 1.0)) / 4.0;
    float errX = sqrt(WaveActiveSum(sumDxSq) / WaveGetLaneCount());
    float errY = sqrt(WaveActiveSum(sumDySq) / WaveGetLaneCount());
    */

    if (all(GroupThreadID.xy == 0))
    {
        nasDataSurface[GroupID.xy] = float2(errX, errY) / abs(avgLuma);
    }
}
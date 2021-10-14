
enum {
    D3D12_SHADING_RATE_1X1 = 0,
    D3D12_SHADING_RATE_1X2 = 0x1,
    D3D12_SHADING_RATE_2X1 = 0x4,
    D3D12_SHADING_RATE_2X2 = 0x5,
    D3D12_SHADING_RATE_2X4 = 0x6,
    D3D12_SHADING_RATE_4X2 = 0x9,
    D3D12_SHADING_RATE_4X4 = 0xa
};

Texture2D<uint> vrsSurface : register(t0);
Texture2D<float2> nasData : register(t1); // for debug vis

#define TILE_SIZE 16

void main_vs(
    in uint iVertex : SV_VertexID,
    out float4 o_posClip : SV_Position)
{
    int u = iVertex & 1;
    int v = (iVertex >> 1) & 1;

    o_posClip = float4(u * 2 - 1, 1 - v * 2, 0, 1);
}


void main_ps(
    in float4 pos : SV_Position,
    out float4 o_rgba : SV_Target)
{
    uint2 xy = uint2(pos.xy);
    uint2 xyGrid = xy % TILE_SIZE;
    uint shadingRate = vrsSurface.Load(uint3(xy / TILE_SIZE, 0));

    float4 overlay = float4(0.0, 0.0, 0.0, 0.0);

    if (shadingRate == D3D12_SHADING_RATE_1X2 || shadingRate == D3D12_SHADING_RATE_2X1)
        overlay = float4(0.0, 0.0, 1.0, 0.3);
    else if (shadingRate == D3D12_SHADING_RATE_2X2)
        overlay = float4(0.0, 1.0, 0.0, 0.3);
    else if (shadingRate == D3D12_SHADING_RATE_2X4 || shadingRate == D3D12_SHADING_RATE_4X2)
        overlay = float4(0.8, 0.8, 0.0, 0.3);
    else if (shadingRate == D3D12_SHADING_RATE_4X4)
        overlay = float4(0.8, 0.0, 0.0, 0.3);

    // White dot if a tile is transposed
    if (shadingRate == D3D12_SHADING_RATE_1X2 || shadingRate == D3D12_SHADING_RATE_2X4)
    {
        if (all(xyGrid > 2) && all(xyGrid < 6))
            overlay = float4(1.0, 1.0, 1.0, 1.0);
    }

    // Tile borders
    if (xyGrid.x == 15 || xyGrid.y == 15)
        overlay = float4(0.0, 0.0, 0.0, 0.5);

    o_rgba = overlay;
}
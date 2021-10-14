
RWTexture2D<uint> vrsSurface : register(u0);

#define TILE_SIZE 16

[numthreads(16, 16, 1)]
void main_cs(uint3 DispatchThreadID : SV_DispatchThreadID, uint3 GroupThreadID : SV_GroupThreadID, uint3 GroupID : SV_GroupID)
{
    uint surfaceWidth, surfaceHeight;
    vrsSurface.GetDimensions(surfaceWidth, surfaceHeight);

    // out-of-bounds check
    if (DispatchThreadID.x >= surfaceWidth || DispatchThreadID.y >= surfaceHeight)
        return;

    uint centerSR = vrsSurface.Load(DispatchThreadID.xy).x;

    // Check all tiles that contain 4x shading rate in either X or Y
    if (centerSR & 0xa)
    {
        bool x1 = false, y1 = false;
        uint SR;
        // check if any of the 4 immediate neighboring tiles has 1x rate
#       define TestOffset(X, Y) \
        SR = vrsSurface.Load(DispatchThreadID.xy + int2(X, Y)); \
        x1 |= ((SR & 0x3) == 0); y1 |= ((SR & 0xc) == 0);

        TestOffset(-1,  0);
        TestOffset( 0, -1);
        TestOffset( 0,  1);
        TestOffset( 1,  0);

        // if an neighboring tile has 1x rate and current tile is 4x in X
        if (x1 && (centerSR & 0x8))
        {
            centerSR ^= 0xc;  // increase the X shading rate from 4x to 2x
        }
        // if an neighboring tile has 1x rate and current tile is 4x in Y
        if (y1 && (centerSR & 0x2))
        {
            centerSR ^= 0x3;  // increase the Y shading rate from 4x to 2x
        }
    }

    vrsSurface[DispatchThreadID.xy] = centerSR;

}

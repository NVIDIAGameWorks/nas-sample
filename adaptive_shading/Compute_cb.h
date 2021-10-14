
#ifndef COMPUTE_CB_H
#define COMPUTE_CB_H

struct ComputeNASDataConstants
{
    float brightnessSensitivity;
};

struct AdaptiveShadingConstants
{
    float4x4 reprojectionMatrix;
    uint2 previousViewOrigin;
    uint2 previousViewSize;
    float2 sourceTextureSizeInv;
    float errorSensitivity;
    float motionSensitivity;
};

#endif // COMPUTE_CB_H
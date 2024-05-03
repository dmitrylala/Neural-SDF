#include "ray_marcher.h"


float3 EyeRayDir(float x, float y, float4x4 a_mViewProjInv)
{
    float4 pos = float4(2.0f * x - 1.0f, 2.0f * y - 1.0f, 0.0f, 1.0f );
    pos = a_mViewProjInv * pos;
    pos /= pos.w;
    return normalize(to_float3(pos));
}


void transform_ray3f(float4x4 a_mWorldViewInv, float3* ray_pos, float3* ray_dir) 
{
    float4 rayPosTransformed = a_mWorldViewInv * to_float4(*ray_pos, 1.0f);
    float4 rayDirTransformed = a_mWorldViewInv * to_float4(*ray_dir, 0.0f);

    (*ray_pos) = to_float3(rayPosTransformed);
    (*ray_dir) = to_float3(normalize(rayDirTransformed));
}


uint32_t RealColorToUint32(float4 real_color)
{
    float r = real_color[0] * 255.0f;
    float g = real_color[1] * 255.0f;
    float b = real_color[2] * 255.0f;
    float a = real_color[3] * 255.0f;

    uint32_t red   = (uint32_t)r;
    uint32_t green = (uint32_t)g;
    uint32_t blue  = (uint32_t)b;
    uint32_t alpha = (uint32_t)a;

    return red | (green << 8) | (blue << 16) | (alpha << 24);
}

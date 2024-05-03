#include "ray_marcher.h"


float sierpinskiSDF(float3 z) {
    int n_iterations = 50;
    float scale = 2.0;
    float offset = 1.0;;

    int n = 0;
    while (n < n_iterations) {
        if (z.x + z.y < 0.0) {
            float tmp = z.x;
            z.x = -z.y;
            z.y = -tmp;
        }

        if (z.x + z.z < 0.0) {
            float tmp = z.x;
            z.x = -z.z;
            z.z = -tmp;
        }

        if (z.y + z.z < 0.0) {
            float tmp = z.z;
            z.z = -z.y;
            z.y = -tmp;
        }
        z = z * scale - offset * (scale - 1.0);
        n++;
    }
    return length(z) * pow(scale, float(-n));
}


float3 EstimateNormal(float3 z)
{  
    float eps = 1e-4;
    float d = sierpinskiSDF(z);
    return normalize(float3(
        sierpinskiSDF(float3(z.x + eps, z.y, z.z)) - d,
        sierpinskiSDF(float3(z.x, z.y + eps, z.z)) - d,
        sierpinskiSDF(float3(z.x, z.y, z.z + eps)) - d
    ));
}

uint32_t MarchOneRay(float3 rayPos, float3 rayDir) {

    int max_iterations = 10000;
    float max_dist = 100.0f;
    float min_dist = 1e-2;

    float3 light = float3(-2.0f, 2.5f, 0.0f);

    float4 resColor(0.0f);
    for (int i = 0; i < max_iterations; ++i) {
        float dist = sierpinskiSDF(rayPos);

        if (dist > max_dist) {
            break;
        }

        float3 new_pos = rayPos + rayDir * dist;

        if (dist <= min_dist) {
            float3 lightDirection = normalize(light - new_pos);
            float3 normal = EstimateNormal(new_pos);
            float color = max(0.1f, dot(lightDirection, normal));
            return RealColorToUint32(float4(color, color, color, 1.0f));
        }

        rayPos = new_pos;
    }

    return RealColorToUint32(resColor);
}

void RayMarcher::kernel2D_RayMarch(uint32_t* out_color, uint32_t width, uint32_t height) 
{
    #ifdef USE_VULKAN
    #else
    #pragma omp parallel for
    #endif
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            float3 rayDir = EyeRayDir((float(x) + 0.5f) / float(width), (float(y) + 0.5f) / float(height), m_worldViewProjInv); 
            float3 rayPos = float3(0.0f, 0.0f, 0.0f);
            transform_ray3f(m_worldViewInv, &rayPos, &rayDir);
            out_color[y * width + x] = MarchOneRay(rayPos, rayDir);
        }
    }
}

void RayMarcher::RayMarch(uint32_t* out_color, uint32_t width, uint32_t height)
{ 
    auto start = std::chrono::high_resolution_clock::now();
    kernel2D_RayMarch(out_color, width, height);
    rayMarchTime = float(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count())/1000.f;
}  

void RayMarcher::GetExecutionTime(const char* a_funcName, float a_out[4])
{
    if (std::string(a_funcName) == "RayMarch")
        a_out[0] = rayMarchTime;
}

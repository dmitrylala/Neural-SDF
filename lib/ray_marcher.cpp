#include "ray_marcher.h"


float3 RayMarcher::EstimateNormal(float3 p) const
{  
    float eps = 1e-4;
    float d = sdf(p);
    return normalize(float3(
        sdf(float3(p.x + eps, p.y, p.z)) - d,
        sdf(float3(p.x, p.y + eps, p.z)) - d,
        sdf(float3(p.x, p.y, p.z + eps)) - d
    ));
}


float3 vecMax(float3 p, float val)
{
    return float3(max(p.x, val), max(p.y, val), max(p.z, val));
}


float unitCubeSDF(float3 p)
{
    float3 d = abs(p) - float3(1.0f, 1.0f, 1.0f);
    return min(max(d.x, max(d.y, d.z)), 0.0f) + length(vecMax(d, 0.0f));
}


std::ostream &operator<<(std::ostream &os, float3 p)
{
    os << p.x << " " << p.y << " " << p.z;
    return os;
}


uint32_t RayMarcher::MarchOneRay(float3 rayPos, float3 rayDir) const
{
    int max_iterations = 100;
    float max_dist = 100.0f;
    float min_dist = 1e-4;

    float4 resColor(0.0f);
    for (int i = 0; i < max_iterations; ++i) {
        float dist = sdf(rayPos);

        if (dist > max_dist) {
            break;
        }

        float3 new_pos = rayPos + rayDir * dist;

        if (dist <= min_dist) {
            float3 lightDirection = normalize(m_light.direction - new_pos);
            float3 normal = EstimateNormal(new_pos);
            float color = max(0.1f, dot(lightDirection, normal)) * m_light.intensity;
            return RealColorToUint32(float4(color, color, color, 1.0f));
        }

        rayPos = new_pos;
    }

    return RealColorToUint32(resColor);
}


float RayMarcher::sdf(float3 p) const
{
    std::vector<float> point = { p.x, p.y, p.z };
    std::vector<float> dist(1);
    m_nn->forward(dist.data(), point.data(), 1);
    return max(dist[0], unitCubeSDF(p));
}


RayMarcher::RayMarcher(Camera cam, Light light, std::shared_ptr<SirenNetwork> net)
{
    const float4x4 view = lookAt(cam.pos, cam.look_at, cam.up);
    const float4x4 proj = perspectiveMatrix(90.0f, 1.0f, cam.z_near, cam.z_far);
    m_worldViewInv      = inverse4x4(view);
    m_worldViewProjInv  = inverse4x4(proj);
    m_nn = net;
    m_light = light;
}


std::vector<uint> RayMarcher::render(uint32_t width, uint32_t height) const
{
    std::vector<uint> out_color(width * height);

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            float3 rayDir = EyeRayDir((float(x) + 0.5f) / float(width), (float(y) + 0.5f) / float(height), m_worldViewProjInv); 
            float3 rayPos = float3(0.0f, 0.0f, 0.0f);
            transform_ray3f(m_worldViewInv, &rayPos, &rayDir);
            out_color[y * width + x] = MarchOneRay(rayPos, rayDir);
        }
    }

    return out_color;
}

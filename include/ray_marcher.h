#pragma once

#include <memory>

#include "siren.h"
#include "configs.h"
#include "utils.h"


class RayMarcher
{
public:
    RayMarcher(Camera cam, Light light, std::shared_ptr<SirenNetwork> net);
    std::vector<uint> render(uint32_t width, uint32_t height) const;

    uint32_t MarchOneRay(float3 rayPos, float3 rayDir) const;
    float3 EstimateNormal(float3 p) const;
    float sdf(float3 p) const;
protected:
    float4x4 m_worldViewProjInv;
    float4x4 m_worldViewInv;
    float    copyTime;
    float    rayMarchTime;
    std::shared_ptr<SirenNetwork> m_nn;
    Light m_light;
};

#pragma once

#include <chrono>
#include <memory>
#include <string>

#include "LiteMath.h"
using namespace LiteMath;


#ifdef USE_VULKAN
#include "vk_context.h"
class RayMarcher;
std::shared_ptr<RayMarcher> CreateRayMarcher_generated(
    vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
#else
#include <omp.h>
#endif


class RayMarcher
{
public:
    RayMarcher()
    {
        // pos, look_at, up
        const float4x4 view = lookAt(float3(0, 1.5, -3), float3(0, 0, 0), float3(0, 1, 0));
        const float4x4 proj = perspectiveMatrix(90.0f, 1.0f, 0.1f, 100.0f);
        m_worldViewInv      = inverse4x4(view); 
        m_worldViewProjInv  = inverse4x4(proj); 
    }

    void SetWorldViewMatrix(const float4x4& a_mat) {m_worldViewInv = inverse4x4(a_mat);}

    virtual void kernel2D_RayMarch(uint32_t* out_color, uint32_t width, uint32_t height);
    virtual void RayMarch(uint32_t* out_color [[size("width*height")]], uint32_t width, uint32_t height);

    virtual void CommitDeviceData() {}
    virtual void UpdateMembersPlainData() {}
    virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);

protected:
    float4x4 m_worldViewProjInv;
    float4x4 m_worldViewInv;
    float    copyTime;
    float    rayMarchTime;
};


std::shared_ptr<RayMarcher> getRayMarcher(uint n_pixels, int n_omp_threads);

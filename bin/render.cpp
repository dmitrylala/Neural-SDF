#include <iostream>
#include <memory>
#include <iomanip>
#include <sstream>

#include "ray_marcher.h"
#include "Image2d.h"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<RayMarcher> CreateRayMarcher_generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
#else
#include <omp.h>
#endif


int main(int argc, const char** argv) {
    #ifndef NDEBUG
    bool enableValidationLayers = true;
    #else
    bool enableValidationLayers = false;
    #endif

    int n_threads = 1;
    if (argc > 1) {
        n_threads = std::max(atoi(argv[1]), 1);
    }

    #ifndef USE_VULKAN
    omp_set_num_threads(n_threads);
    #endif

    uint WIN_WIDTH  = 512;
    uint WIN_HEIGHT = 512;

    std::shared_ptr<RayMarcher> pImpl = nullptr;
    #ifdef USE_VULKAN
    bool onGPU = true; // TODO: you can read it from command line
    if(onGPU)
    {
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, 0);
    pImpl    = CreateRayMarcher_generated(ctx, WIN_WIDTH*WIN_HEIGHT);
    }
    else
    #else
    bool onGPU = false;
    #endif
    pImpl = std::make_shared<RayMarcher>();

    pImpl->CommitDeviceData();

    std::vector<uint> pixelData(WIN_WIDTH*WIN_HEIGHT);


    float4x4 mRotX = rotate4x4X(float(-25) * DEG_TO_RAD);
    float4 camPos = mRotX * float4(1, 1, 2, 1);

    // pos, look_at, up
    float4x4 viewMat = lookAt(to_float3(camPos), float3(0, 0, 0), float3(0, 1, 0));

    pImpl->SetWorldViewMatrix(viewMat);
    pImpl->UpdateMembersPlainData();
    pImpl->RayMarch(pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

    float timings[4] = {0, 0, 0, 0};
    pImpl->GetExecutionTime("RayMarch", timings);

    std::stringstream strOut;
    if (onGPU)
        strOut << std::fixed << std::setprecision(2) << "out_gpu.bmp";
    else
        strOut << std::fixed << std::setprecision(2) << "out_cpu.bmp";
    std::string fileName = strOut.str();

    LiteImage::SaveBMP(fileName.c_str(), pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

    std::cout << "n_threads = " << n_threads << ", onGPU = " << onGPU << std::endl;
    std::cout << "timeRender = " << timings[0] << " ms, timeCopy = " <<  timings[1] + timings[2] << " ms " << std::endl;

    pImpl = nullptr;
    return 0;
}
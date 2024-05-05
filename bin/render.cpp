#include <iostream>
#include <memory>
#include <iomanip>
#include <sstream>

#include "ray_marcher.h"
#include "Image2d.h"

#ifdef USE_VULKAN
static const bool onGPU = true;
#else
static const bool onGPU = false;
#endif



int main(int argc, const char** argv) {
    int n_omp_threads = 1;
    if (argc > 1) {
        n_omp_threads = std::max(atoi(argv[1]), 1);
    }

    uint WIN_WIDTH  = 512;
    uint WIN_HEIGHT = 512;

    auto ray_marcher = getRayMarcher(WIN_WIDTH * WIN_HEIGHT, n_omp_threads);
    ray_marcher->CommitDeviceData();

    std::vector<uint> pixelData(WIN_WIDTH*WIN_HEIGHT);

    float4x4 mRotX = rotate4x4X(float(-25) * DEG_TO_RAD);
    float4 camPos = mRotX * float4(1, 1, 2, 1);

    // pos, look_at, up
    float4x4 viewMat = lookAt(to_float3(camPos), float3(0, 0, 0), float3(0, 1, 0));

    ray_marcher->SetWorldViewMatrix(viewMat);
    ray_marcher->UpdateMembersPlainData();
    ray_marcher->RayMarch(pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

    float timings[4] = {0, 0, 0, 0};
    ray_marcher->GetExecutionTime("RayMarch", timings);

    std::stringstream strOut;
    if (onGPU)
        strOut << std::fixed << std::setprecision(2) << "out_gpu.bmp";
    else
        strOut << std::fixed << std::setprecision(2) << "out_cpu.bmp";

    std::string fileName = strOut.str();

    LiteImage::SaveBMP(fileName.c_str(), pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

    std::cout << "n_threads = " << n_omp_threads << ", onGPU = " << onGPU << std::endl;
    std::cout << "timeRender = " << timings[0] << " ms, timeCopy = " <<  timings[1] + timings[2] << " ms " << std::endl;

    ray_marcher = nullptr;
    return 0;
}
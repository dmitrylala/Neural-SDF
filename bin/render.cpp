#include <iostream>
#include <memory>

#include "Image2d.h"
#include "argparser.h"

#include "ray_marcher.h"
#include "utils.h"

#ifdef USE_VULKAN
static const bool onGPU = true;
#else
static const bool onGPU = false;
#endif


static const int WIN_HEIGHT = 512;
static const int WIN_WIDTH = 512;



int main(int argc, const char** argv)
{
    ArgParser parser(argc, argv);
    const int n_omp_threads = parser.get_n_threads();
    const std::vector<float> weights = parse_weights(parser);

    std::cout << "Running with n_threads = " << n_omp_threads << ", onGPU = " \
        << onGPU << std::endl;

    auto ray_marcher = getRayMarcher(WIN_WIDTH * WIN_HEIGHT, n_omp_threads);
    auto net = getSirenNetwork(2, 64, 10);
    net->setWeights(weights);
    net->CommitDeviceData();
    ray_marcher->SetSDFNetwork(net);
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
    std::cout << "Rendering done, timeRender = " << timings[0] << " ms, timeCopy = " \
        <<  timings[1] + timings[2] << " ms " << std::endl;

    std::string save_to;
    if (onGPU)
        save_to = "out_gpu.bmp";
    else
        save_to = "out_cpu.bmp";
    LiteImage::SaveBMP(save_to.c_str(), pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

    ray_marcher = nullptr;
    return 0;
}
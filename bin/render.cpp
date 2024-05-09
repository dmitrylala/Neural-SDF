#include <iostream>
#include <memory>
#include <chrono>

#include "Image2d.h"
#include "argparser.h"

#include "ray_marcher.h"

#ifdef USE_VULKAN
static const bool onGPU = true;
#else
static const bool onGPU = false;
#endif


static const int DEFAULT_RES = 512;



int main(int argc, const char** argv)
{
    ArgParser parser(argc, argv);

    const int resolution = parser.getOptionValue<int>("--resolution", DEFAULT_RES);

    Camera cam = load_cam(parser.getOptionValue<std::string>("--camera"));
    Light light = load_light(parser.getOptionValue<std::string>("--light"));

    const auto [n_hidden_layers, hidden_size, batch_size] = parser.get_network_setup();
    const auto weights = load_floats(parser.getOptionValue<std::string>("--weights"));

    auto net = getSirenNetwork(n_hidden_layers, hidden_size, batch_size);
    net->setWeights(weights);
    net->CommitDeviceData();

    auto ray_marcher = RayMarcher(cam, light, net);

    std::cout << "Rendering with resolution: " << resolution << \
        ", on GPU: " << onGPU << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<uint> pixelData = ray_marcher.render(resolution, resolution);
    float renderTime = float(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count()) / 1000.f;
    std::cout << "Render done, elapsed = " << renderTime << " ms" << std::endl;

    std::string save_to;
    if (onGPU)
        save_to = "out_gpu.bmp";
    else
        save_to = "out_cpu.bmp";
    LiteImage::SaveBMP(save_to.c_str(), pixelData.data(), resolution, resolution);

    return 0;
}
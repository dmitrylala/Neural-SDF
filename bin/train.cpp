#include <iostream>
#include <memory>

#include "siren.h"
#include "utils.h"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<SirenNetwork> CreateSirenNetwork_generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
#endif


static const bool enableValidationLayers = false;
static const int N_COORDS = 3;



int main(int argc, const char** argv) {
    ArgParser parser(argc, argv);

    const auto [n_hidden_layers, hidden_size, batch_size] = parse_network_setup(parser);

    const std::vector<float> weights = parse_weights(parser);
    const auto [points, gt_sdf] = parse_test_points(parser);

    std::cout << "First batch_size gt sdfs:" << std::endl;
    for (int i = 0; i < batch_size; ++i)
        std::cout << gt_sdf[i] << " ";
    std::cout << std::endl;


    std::shared_ptr<SirenNetwork> pImpl = nullptr;
    #ifdef USE_VULKAN
    bool onGPU = true;
    if (onGPU) {
        auto ctx = vk_utils::globalContextGet(enableValidationLayers, 0);
        pImpl = CreateSirenNetwork_generated(ctx, batch_size);
    } else
    #else  
        bool onGPU = false;
    #endif
    pImpl = std::make_shared<SirenNetwork>();

    pImpl->init(n_hidden_layers, hidden_size, batch_size, N_COORDS, 1);
    if (weights.size() > 0) {
        pImpl->setWeights(weights);
        std::cout << "Loaded weights: " << weights.size() << std::endl;
    }

    pImpl->CommitDeviceData();

    std::cout << "Build succeeded" << std::endl;

    std::vector<float> pred_sdf(batch_size);

    std::vector<float> points_batch(batch_size * N_COORDS);
    int batch_offset = 0;

    int i = 0;
    for (int row = batch_offset; row < batch_offset + batch_size; ++row) {
        for (int col = 0; col < N_COORDS; ++col) {
            points_batch[i] = points[row * N_COORDS + col];
            ++i;
        }
    }
    points_batch = transpose(points_batch, batch_size, N_COORDS);

    pImpl->UpdateMembersPlainData();
    pImpl->forward(pred_sdf.data(), points_batch.data());

    std::cout << "Forward done" << std::endl;

    std::cout << "First batch_size predicted sdf values:" << std::endl;
    for (int i = 0; i < batch_size; ++i) {
        std::cout << pred_sdf[i] << " ";
    }
    std::cout << std::endl;

    pImpl = nullptr;
    return 0;
}

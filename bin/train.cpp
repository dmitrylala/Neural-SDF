#include <iostream>
#include <memory>
#include <chrono>

#include "siren.h"
#include "utils.h"



int main(int argc, const char** argv)
{
    ArgParser parser(argc, argv);

    const auto [n_hidden_layers, hidden_size, batch_size] = parser.get_network_setup();
    std::cout << "Network setup: n_hidden = " << n_hidden_layers << \
        ", hidden_size = " << hidden_size << ", batch_size = " << batch_size << std::endl;

    const std::vector<float> init_weights = parse_weights(parser);
    const auto [points, gt_sdf] = parse_test_points(parser);

    std::cout << "First batch_size gt sdfs:" << std::endl;
    for (int i = 0; i < batch_size; ++i)
        std::cout << gt_sdf[i] << " ";
    std::cout << std::endl;

    auto net = getSirenNetwork(n_hidden_layers, hidden_size, batch_size);
    if (init_weights.size() > 0) {
        net->setWeights(init_weights);
        std::cout << "Loaded weights: " << init_weights.size() << std::endl;
    }

    net->CommitDeviceData();

    std::vector<float> pred_sdf(batch_size);

    std::vector<float> points_batch(batch_size * INPUT_DIM);
    int batch_offset = 0;

    int i = 0;
    for (int row = batch_offset; row < batch_offset + batch_size; ++row) {
        for (int col = 0; col < INPUT_DIM; ++col) {
            points_batch[i] = points[row * INPUT_DIM + col];
            ++i;
        }
    }
    points_batch = transpose(points_batch, batch_size, INPUT_DIM);

    net->UpdateMembersPlainData();

    int k = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < k; ++i) {
        net->forward(pred_sdf.data(), points_batch.data());
    }
    auto elapsed = float(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count())/1000.f;

    std::cout << k << " times forward done, elapsed = " << elapsed << std::endl;

    std::cout << "First batch_size predicted sdf values:" << std::endl;
    for (int i = 0; i < batch_size; ++i) {
        std::cout << pred_sdf[i] << " ";
    }
    std::cout << std::endl;

    net = nullptr;
    return 0;
}

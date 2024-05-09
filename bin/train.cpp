#include <iostream>
#include <chrono>

#include "siren.h"
#include "argparser.h"
#include "utils.h"
#include "configs.h"



int main(int argc, const char** argv)
{
    ArgParser parser(argc, argv);

    const auto [n_hidden_layers, hidden_size, batch_size] = parser.get_network_setup();
    std::cout << "Network setup: n_hidden = " << n_hidden_layers << \
        ", hidden_size = " << hidden_size << ", batch_size = " << batch_size << std::endl;

    const auto [points, sdfs] = load_points(
        parser.getOptionValue<std::string>("--train_sample"));

    const auto train_cfg = load_train_cfg(parser.getOptionValue<std::string>("--train_cfg"));

    const std::string save_to = parser.getOptionValue<std::string>("--save_to");

    auto net = getSirenNetwork(n_hidden_layers, hidden_size, batch_size);
    net->CommitDeviceData();
    net->UpdateMembersPlainData();

    int n_batches = (sdfs.size() + batch_size) / batch_size;
    std::vector<std::vector<float>> x_batches = batchify(points, batch_size, n_batches, 3);
    std::vector<std::vector<float>> y_batches = batchify(sdfs, batch_size, n_batches, 1);

    std::cout << "Running train with lr: " << train_cfg.lr << ", n_epochs: " << \
        train_cfg.n_epochs << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < train_cfg.n_epochs; ++epoch) {
        auto batch_idxs = shuffle_batch_idxs(n_batches);

        std::vector<float> epoch_losses(n_batches);
        for (auto batch_idx: batch_idxs) {
            auto x_batch = x_batches[batch_idx];
            auto y_batch = y_batches[batch_idx];
            std::vector<float> preds(y_batch.size());

            net->forward(preds.data(), x_batch.data(), y_batch.size());

            float loss = mse_loss(preds, y_batch);
            epoch_losses[batch_idx] = loss;
            net->backward(y_batch.data());
            net->step(train_cfg.lr);
        }

        float mean_epoch_loss = 0.0f;
        for (auto loss: epoch_losses)
            mean_epoch_loss += loss;
        mean_epoch_loss /= n_batches;

        if (epoch % train_cfg.log_every_n_epochs == 0)
            std::cout << "Epoch: " << epoch << ", loss: " << mean_epoch_loss << std::endl;
    }

    auto elapsed = float(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count()) / 1000.f;
    std::cout << "Training finished, elapsed = " << elapsed << " ms" << std::endl;

    auto weights = net->getWeights();
    std::ofstream fout(save_to, std::ios::out | std::ios::binary);
    fout.write((char*)&weights[0], weights.size() * sizeof(float));
    fout.close();
    std::cout << "Saved weights to = " << save_to << std::endl;

    net = nullptr;
    return 0;
}

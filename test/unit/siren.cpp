#include <catch2/catch_test_macros.hpp>

#include <iostream>

#include "siren.h"
#include "utils.h"


TEST_CASE( "matmul", "[siren]" )
{
    // architecture arguments don't matter
    auto net = getSirenNetwork(2, 64, 10);

    std::vector<float> a = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };

    std::vector<float> b = {
        1.0, 2.0,
        4.0, 5.0,
        6.0, 7.0
    };

    std::vector<float> c(2 * 2);

    std::vector<float> c_gt = {
        27, 33,
        60, 75
    };

    net->kernel2D_matmul(
        c.data(), a.data(), b.data(),
        // shapes
        2, 3, 2
    );

    REQUIRE( c_gt == c );
}


TEST_CASE( "add bias", "[siren]" )
{
    // architecture arguments don't matter
    auto net = getSirenNetwork(2, 64, 10);

    std::vector<float> matrix = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };

    std::vector<float> bias = {
        1.0, -3.0
    };

    std::vector<float> res(3 * 2);

    std::vector<float> res_gt = {
        2.0, 3.0, 4.0,
        1.0, 2.0, 3.0
    };

    net->kernel2D_add_bias(
        res.data(), matrix.data(), bias.data(),
        // shapes
        2, 3
    );

    REQUIRE( res_gt == res );
}


TEST_CASE( "forward on test points", "[siren]" )
{
    const std::vector<float> init_weights = read_weights("data/sdf1_weights.bin");
    const auto [points, gt_sdf] = read_test_points("data/sdf1_test.bin");

    const int batch_size = gt_sdf.size();

    auto net = getSirenNetwork(2, 64, batch_size);

    std::vector<float> pred_sdf(batch_size);

    net->setWeights(init_weights);
    net->CommitDeviceData();

    std::vector<float> points_batch = transpose(points, batch_size, 3);

    net->UpdateMembersPlainData();
    net->forward(pred_sdf.data(), points_batch.data());

    float test_mse = 0.0f;
    for (int i = 0; i < pred_sdf.size(); ++i) {
        float diff = pred_sdf[i] - gt_sdf[i];
        test_mse += diff * diff;
    }
    test_mse /= pred_sdf.size();

    int first_k = 10;
    std::cout << "First " << first_k << " values in gt sdfs:" << std::endl;
    print_first_k(std::cout, gt_sdf, first_k);

    std::cout << "First " << first_k << " values in pred sdfs:" << std::endl;
    print_first_k(std::cout, pred_sdf, first_k);

    std::cout << "Forward test MSE = " << test_mse << std::endl;

    REQUIRE( test_mse < 1e-9f );
}

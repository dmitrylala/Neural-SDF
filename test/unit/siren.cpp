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
    const std::vector<float> init_weights = load_floats("data/sdf1_weights.bin");
    const auto [points, gt_sdf] = load_points("data/sdf1_test.bin");

    const int batch_size = gt_sdf.size();

    auto net = getSirenNetwork(2, 64, batch_size);

    std::vector<float> pred_sdf(batch_size);

    net->setWeights(init_weights);
    net->CommitDeviceData();

    std::vector<float> points_batch = transpose(points, batch_size, 3);

    net->UpdateMembersPlainData();
    net->forward(pred_sdf.data(), points_batch.data(), batch_size);

    float test_mse = mse_loss(pred_sdf, gt_sdf);

    int first_k = 10;
    std::cout << "[Forward] First " << first_k << " values in gt sdfs:" << std::endl;
    print_first_k(std::cout, gt_sdf, first_k);

    std::cout << "[Forward] First " << first_k << " values in pred sdfs:" << std::endl;
    print_first_k(std::cout, pred_sdf, first_k);

    std::cout << "[Forward] Forward test MSE = " << test_mse << std::endl;

    REQUIRE( test_mse < 1e-9f );
}


TEST_CASE( "backward on sampled batch", "[siren]" )
{
    const std::vector<float> weights = {
        -0.020675774468495136, 0.02453392708879327, 0.016319536795603974,
        0.024981532236227448, 0.017221389215200646, -0.0457476239651699,
        0.0, 0.0,
        
        -0.03169821925648701, 0.04923352805229247,
        -0.036622258170210156, 0.015040030468648345,
        0.0, 0.0,
        
        -0.03312720234831627, -0.02018968030203111,
        -0.049434935925399465, 0.029500214575452745,
        0.0, 0.0,
        
        -0.0323701214895121,
        -0.003361059539908194,
        0.0
    };

    const std::vector<float> x_batch_transposed = {
        -0.270431, 0.83239,
        0.0268018, 0.271423,
        0.904459, 0.434594
    };
    const std::vector<float> y_batch = {
        0.234232, 0.433599
    };
    const int batch_size = y_batch.size();

    auto net = getSirenNetwork(2, 2, batch_size);
    net->setWeights(weights);
    net->CommitDeviceData();

    std::vector<float> preds(batch_size);
    net->UpdateMembersPlainData();
    net->forward(preds.data(), x_batch_transposed.data(), batch_size);
    
    // forward
    const std::vector<float> preds_gt = { -0.03380972, 0.0151809 };
    float mse = mse_loss(preds, preds_gt);
    REQUIRE( mse < 1e-5 );

    // mse loss
    const float gt_loss = 0.12346003756881406;
    float loss = mse_loss(preds, y_batch);
    REQUIRE( abs(gt_loss - loss) < 1e-9 );

    net->backward(y_batch.data());

    auto w_grads = net->getWeightsGradients();
    auto out_grads = net->getOutputsGradients();

    // mse gradient
    const std::vector<float> gt_mse_grads = { -0.26804172, -0.4184181 };
    std::vector<float> mse_grads = {
        out_grads[out_grads.size() - 2],
        out_grads[out_grads.size() - 1]
    };
    mse = mse_loss(mse_grads, gt_mse_grads);
    REQUIRE( mse < 1e-9 );

    // weights gradients
    const std::vector<float> gt_w_grads = {
        0.4859084199034893, 0.1558075211890458, 0.22972383758595288,
        -0.5517433906864425, -0.1791875277352366, -0.2814910603510409,
        0.553378895177955, -0.6545099937233934,

        0.059219764274264286, -0.09662933335821602,
        0.018591181023966383, -0.029917637430147193,
        -0.3672719634987714, -0.1836654887395775,

        0.0913628917424316, 0.03681359251935215,
        -0.007285127299885221, -0.013015265829755021,
        0.4018658514231768, 0.06291764445068225,

        -0.08352910186967984,
        -0.0019659154657334925,
        -0.6864598270845845
    };
    mse = mse_loss(w_grads, gt_w_grads);
    std::cout << "[Backward small] Weights gradients MSE: " << mse << std::endl;
    REQUIRE( mse < 1e-9 );
}


TEST_CASE( "backward on bigger sampled batch", "[siren]" )
{
    const std::vector<float> weights = load_floats("data/test_unit/weights.bin");

    auto [x_batch, y_batch] = load_points("data/test_unit/points.bin");
    const int batch_size = y_batch.size();
    x_batch = transpose(x_batch, batch_size, INPUT_DIM);

    auto net = getSirenNetwork(2, 10, batch_size);
    net->setWeights(weights);
    net->CommitDeviceData();

    std::vector<float> preds(batch_size);
    net->UpdateMembersPlainData();
    net->forward(preds.data(), x_batch.data(), batch_size);

    // forward
    const auto preds_gt = load_floats("data/test_unit/preds.bin");
    float mse = mse_loss(preds, preds_gt);
    REQUIRE( mse < 1e-5 );

    // mse loss
    const float gt_loss = load_floats("data/test_unit/loss.bin")[0];
    float loss = mse_loss(preds, y_batch);
    REQUIRE( abs(gt_loss - loss) < 1e-7 );

    net->backward(y_batch.data());

    auto w_grads = net->getWeightsGradients();
    auto out_grads = net->getOutputsGradients();

    const auto gt_mse_grads = load_floats("data/test_unit/loss_grads.bin");
    const auto gt_w_grads = load_floats("data/test_unit/weights_grads.bin");

    // mse gradient
    std::vector<float> mse_grads;
    for (int i = batch_size; i >= 1; --i) {
        mse_grads.push_back(out_grads[out_grads.size() - i]);
    };
    mse = mse_loss(mse_grads, gt_mse_grads);
    std::cout << "[Backward] MSE gradients MSE: " << mse << std::endl;
    REQUIRE( mse < 1e-9 );

    // weights gradients
    mse = mse_loss(w_grads, gt_w_grads);
    std::cout << "[Backward] Weights gradients MSE: " << mse << std::endl;
    REQUIRE( mse < 1e-9 );
}


TEST_CASE( "Adam step on sampled batch", "[siren]" )
{
    const std::vector<float> weights = {
        0.04071933506347348, 0.017937196236156062, 0.014314661404559309,
        0.011497761463557021, -0.025524037854656498, 0.019999861020300345,
        0.0, 0.0,
        
        -0.05075432008643307, -0.009519573271178472,
        0.04207575638162301, -0.019114603326144654,
        0.0, 0.0,
        
        0.011140624125125252, 0.0011878291330219698,
        0.01577979444881058, -0.049033644845920515,
        0.0, 0.0,
        
        0.05430067829836938,
        -0.039086576887520506,
        0.0
    };

    const std::vector<float> x_batch_transposed = {
        0.14857023, 0.72472809,
        0.78884354, 0.81606277,
        0.19605531, 0.44920772
    };
    const std::vector<float> y_batch = {
        0.66931605, 0.01862813
    };
    const int batch_size = y_batch.size();

    auto net = getSirenNetwork(2, 2, batch_size);
    net->setWeights(weights);
    net->CommitDeviceData();

    std::vector<float> preds(batch_size);
    net->UpdateMembersPlainData();
    net->forward(preds.data(), x_batch_transposed.data(), batch_size);
    
    net->backward(y_batch.data());
    net->step(5e-5);

    const std::vector<float> gt_updated_weights = {
        0.04066933507494174, 0.01788719623831511, 0.014264661413247545,
        0.011447761524204637, -0.02557403784312928, 0.01994986106654213,
        -4.999999829685806e-05, -4.9999990906078836e-05,
        
        -0.050704320089723594, -0.009569573266212494,
        0.04202575644194964, -0.019064603414586608,
        4.999999790569382e-05, -4.9999962085009174e-05,
        
        0.011090624125780968, 0.0012378291324534648,
        0.01572979446474642, -0.04898364485968525,
        4.9999999508394724e-05, 4.9999988122755575e-05,
        
        0.05425067830195481,
        -0.039136576886737794,
        4.999999921841459e-05
    };
    auto updated_weights = net->getWeights();

    float mse = mse_loss(updated_weights, gt_updated_weights);
    std::cout << "[Adam step] Updated weights after Adam step MSE: " << mse << std::endl;
    REQUIRE( mse < 1e-9 );
}

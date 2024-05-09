#pragma once

#include <vector>
#include <utility>
#include <cmath>
#include <memory>
#include <random>


#ifdef USE_VULKAN
#include "vk_context.h"
class SirenNetwork;
std::shared_ptr<SirenNetwork> CreateSirenNetwork_generated(
    int n_hidden, int hidden_size, int batch_size,
    vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
#endif


static const int INPUT_DIM = 3;
static const int OUTPUT_DIM = 1;


class SirenNetwork
{
public:
    SirenNetwork(int n_hidden, int hidden_size, int batch_size);
    void setWeights(const std::vector<float> &weights);
    std::vector<float> getWeights() const;

    // for testing purposes
    std::vector<float> getWeightsGradients() const;
    std::vector<float> getOutputsGradients() const;

    void forward(float *res, const float *input, int batch_size);
    void backward(const float *y_gt);
    void step(float lr);

    void kernel2D_matmul(
        float *c, float *a, float *b,
        uint32_t a_rows, uint32_t b_rows, uint32_t b_cols,
        uint32_t c_offset = 0, uint32_t a_offset = 0, uint32_t b_offset = 0);
    void kernel2D_add_bias(
        float *res, float *inp, float *vec,
        uint32_t n_rows, uint32_t n_cols,
        uint32_t res_offset = 0, uint32_t input_offset = 0, uint32_t vec_offset = 0);
    void kernel2D_sin_activation(
        float *res, float *inp,
        uint32_t n_rows, uint32_t n_cols,
        uint32_t res_offset, uint32_t input_offset);

    void kernel1D_mse_grad(
        float *res, float *preds, float *gt,
        uint32_t n_samples,
        uint32_t res_offset = 0, uint32_t preds_offset = 0, uint32_t gt_offset = 0);
    void kernel2D_bias_grad(
        float *res, float *inp,
        uint32_t n_rows, uint32_t n_cols,
        uint32_t res_offset = 0, uint32_t input_offset = 0);
    void kernel2D_sin_grad(
        float *res, float *inp, float *out_grads,
        uint32_t n_rows, uint32_t n_cols,
        uint32_t res_offset = 0, uint32_t input_offset = 0, uint32_t out_grads_offset = 0);

    // for backward
    void kernel2D_matmul_transposed_right(
        float *c, float *a, float *b,
        uint32_t a_rows, uint32_t b_rows, uint32_t b_cols,
        uint32_t c_offset = 0, uint32_t a_offset = 0, uint32_t b_offset = 0);
    void kernel2D_matmul_transposed_left(
        float *c, float *a, float *b,
        uint32_t a_rows, uint32_t b_rows, uint32_t b_cols,
        uint32_t c_offset = 0, uint32_t a_offset = 0, uint32_t b_offset = 0);

    void kernel1D_Adam_step(
        float *params, float *grads, float *adam_m, float *adam_v,
        uint32_t n_params, float lr);


    virtual void UpdateMembersPlainData() {}
    virtual void CommitDeviceData() {}
protected:
    std::vector<float> m_weights_biases, m_weights_grads, m_outputs, m_out_grads;
    std::vector<std::pair<int,int>> m_layers_shapes;
    int m_batch_size, m_outputs_end;
    
    // for copying y_gt batch for loss computation
    std::vector<float> m_gt_buffer;

    // Adam optimizer
    // grad momentums
    std::vector<float> m_adam_m, m_adam_v;

    // betas and steps counter
    float beta1 = 0.9, beta2 = 0.99, eps = 1e-8;
    int t = 1;
};


std::shared_ptr<SirenNetwork> getSirenNetwork(int n_hidden, int hidden_size, int batch_size);

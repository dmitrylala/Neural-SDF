#pragma once

#include <vector>
#include <utility>
#include <cmath>
#include <cstdint>
#include <memory>


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
    void forward(float *res, float *input);

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


    virtual void UpdateMembersPlainData() {}
    virtual void CommitDeviceData() {}
protected:
    std::vector<float> m_layers_outputs;
    std::vector<float> m_weights_biases;
    std::vector<std::pair<int,int>> m_layers_shapes;
    int m_batch_size;
};


std::shared_ptr<SirenNetwork> getSirenNetwork(int n_hidden, int hidden_size, int batch_size);

#pragma once

#include <vector>
#include <utility>
#include <cmath>
#include <cstdint>


class SirenNetwork
{
public:
    SirenNetwork() {}

    void init(int n_hidden, int hidden_size, int batch_size, int input_dim, int out_dim);
    void setWeights(const std::vector<float> &weights);

    void forward(float *res, float *input);


    void kernel2D_matmul(
        float *c, float *a, float *b,
        uint32_t a_rows, uint32_t b_rows, uint32_t b_cols,
        uint32_t c_offset, uint32_t a_offset, uint32_t b_offset);
    void kernel2D_add_bias(
        float *res, float *inp, float *vec,
        uint32_t n_rows, uint32_t n_cols,
        uint32_t res_offset, uint32_t input_offset, uint32_t vec_offset);
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
#include "siren.h"


void SirenNetwork::kernel2D_matmul(
    float *c, float *a, float *b,
    uint32_t a_rows, uint32_t b_rows, uint32_t b_cols,
    uint32_t c_offset, uint32_t a_offset, uint32_t b_offset)
{
    for (uint32_t i = 0; i < a_rows; ++i) {
        for (uint32_t j = 0; j < b_cols; ++j) {
            float value = 0.0f;
            for (uint32_t k = 0; k < b_rows; ++k) {
                value += a[a_offset + i * b_rows + k] * b[b_offset + k * b_cols + j];
            }
            c[c_offset + i * b_cols + j] = value;
        }
    }
}


void SirenNetwork::kernel2D_add_bias(
    float *res, float *inp, float *vec,
    uint32_t n_rows, uint32_t n_cols,
    uint32_t res_offset, uint32_t input_offset, uint32_t vec_offset)
{
    for (uint32_t i = 0; i < n_rows; ++i) {
        for (uint32_t j = 0; j < n_cols; ++j) {
            res[res_offset + i * n_cols + j] = inp[input_offset + i * n_cols + j] + \
                                            vec[vec_offset + i];
        }
    }
}


void SirenNetwork::kernel2D_sin_activation(
    float *res, float *inp,
    uint32_t n_rows, uint32_t n_cols,
    uint32_t res_offset, uint32_t input_offset)
{
    for (uint32_t i = 0; i < n_rows; ++i) {
        for (uint32_t j = 0; j < n_cols; ++j) {
            float x = 30.0 * inp[input_offset + i * n_cols + j];
            res[res_offset + i * n_cols + j] = sin(x);
        }
    }
}


void SirenNetwork::init(
    int n_hidden, int hidden_size, int batch_size, int input_dim, int out_dim)
{
    m_batch_size = batch_size;

    m_layers_shapes.push_back(std::pair<int,int>{hidden_size, input_dim});
    for (int i = 0; i < n_hidden; ++i) {
        m_layers_shapes.push_back(std::pair<int,int>{hidden_size, hidden_size});
    }
    m_layers_shapes.push_back(std::pair<int,int>{out_dim, hidden_size});

    // input will be copied to outputs attr
    int n_params = 0, n_outputs = m_batch_size * input_dim;

    for (auto [out_dim, in_dim]: m_layers_shapes) {
        n_params += in_dim * out_dim + out_dim;
        // two outputs for linear layer and one for sin activation
        n_outputs += 3 * m_batch_size * out_dim;
    }
    // there is no activation after last linear layer
    n_outputs -= m_batch_size * m_layers_shapes.back().first;

    m_weights_biases = std::vector<float>(n_params);
    m_layers_outputs = std::vector<float>(n_outputs);

    // TODO: add weights initialization here
}


void SirenNetwork::setWeights(const std::vector<float> &weights)
{
    m_weights_biases = weights;
}


void SirenNetwork::forward(float *res, float *input)
{
    int first_in_dim = m_layers_shapes.front().second;
    for (int i = 0; i < m_batch_size * first_in_dim; ++i) {
        m_layers_outputs[i] = input[i];
    }

    uint32_t w_offset = 0, out_offset = m_batch_size * first_in_dim, in_offset = 0;
    int layer_i = 0;
    for (auto [out_dim, in_dim]: m_layers_shapes) {
        kernel2D_matmul(
            m_layers_outputs.data(), m_weights_biases.data(), m_layers_outputs.data(),
            out_dim, in_dim, m_batch_size,
            out_offset, w_offset, in_offset);
        in_offset = out_offset;
        out_offset += m_batch_size * out_dim;
        w_offset += in_dim * out_dim;

        kernel2D_add_bias(
            m_layers_outputs.data(), m_layers_outputs.data(), m_weights_biases.data(),
            out_dim, m_batch_size,
            out_offset, in_offset, w_offset);
        in_offset = out_offset;
        out_offset += m_batch_size * out_dim;
        w_offset += out_dim;

        if (layer_i < m_layers_shapes.size() - 1) {
            kernel2D_sin_activation(
                m_layers_outputs.data(), m_layers_outputs.data(),
                m_batch_size, out_dim,
                out_offset, in_offset);
            in_offset = out_offset;
            out_offset += m_batch_size * out_dim;
        }

        ++layer_i;
    }

    int last_out_dim = m_layers_shapes.back().first;
    int res_offset = m_layers_outputs.size() - m_batch_size * last_out_dim;
    for (int i = 0; i < m_batch_size * last_out_dim; ++i) {
        res[i] = m_layers_outputs[res_offset + i];
    }
}

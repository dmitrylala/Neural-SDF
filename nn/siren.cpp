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


void SirenNetwork::kernel1D_mse_grad(
    float *res, float *preds, float *gt,
    uint32_t n_samples,
    uint32_t res_offset, uint32_t preds_offset, uint32_t gt_offset)
{
    for (uint32_t i = 0; i < n_samples; ++i) {
        res[res_offset + i] = 2 * (preds[preds_offset + i] - gt[gt_offset + i]) / n_samples;
    }
}


void SirenNetwork::kernel2D_bias_grad(
    float *res, float *inp,
    uint32_t n_rows, uint32_t n_cols,
    uint32_t res_offset, uint32_t input_offset)
{
    for (uint32_t i = 0; i < n_rows; ++i) {
        for (uint32_t j = 0; j < n_cols; ++j) {
            res[res_offset + i] += inp[input_offset + i * n_cols + j];
        }
    }
}


void SirenNetwork::kernel2D_matmul_transposed_right(
    float *c, float *a, float *b,
    uint32_t a_rows, uint32_t b_rows, uint32_t b_cols,
    uint32_t c_offset, uint32_t a_offset, uint32_t b_offset)
{
    for (uint32_t i = 0; i < a_rows; ++i) {
        for (uint32_t j = 0; j < b_cols; ++j) {
            float value = 0.0f;
            for (uint32_t k = 0; k < b_rows; ++k) {
                value += a[a_offset + i * b_rows + k] * b[b_offset + j * b_rows + k];
            }
            c[c_offset + i * b_cols + j] = value;
        }
    }
}


void SirenNetwork::kernel2D_matmul_transposed_left(
    float *c, float *a, float *b,
    uint32_t a_rows, uint32_t b_rows, uint32_t b_cols,
    uint32_t c_offset, uint32_t a_offset, uint32_t b_offset)
{
    for (uint32_t i = 0; i < a_rows; ++i) {
        for (uint32_t j = 0; j < b_cols; ++j) {
            float value = 0.0f;
            for (uint32_t k = 0; k < b_rows; ++k) {
                value += a[a_offset + k * a_rows + i] * b[b_offset + k * b_cols + j];
            }
            c[c_offset + i * b_cols + j] = value;
        }
    }
}


void SirenNetwork::kernel2D_sin_grad(
    float *res, float *inp, float *out_grads,
    uint32_t n_rows, uint32_t n_cols,
    uint32_t res_offset, uint32_t input_offset, uint32_t out_grads_offset)
{
    for (uint32_t i = 0; i < n_rows; ++i) {
        for (uint32_t j = 0; j < n_cols; ++j) {
            uint32_t idx = i * n_cols + j;
            float x_cos = cos(30.0 * inp[input_offset + idx]);
            res[res_offset + idx] = 30.0 * x_cos * out_grads[out_grads_offset + idx];
        }
    }
}


void SirenNetwork::kernel1D_Adam_step(
    float *params, float *grads, float *adam_m, float *adam_v, uint32_t n_params, float lr)
{
    for (uint32_t i = 0; i < n_params; ++i) {
        adam_m[i] = beta1 * adam_m[i] + (1 - beta1) * grads[i];
        adam_v[i] = beta2 * adam_v[i] + (1 - beta2) * pow(grads[i], 2);

        float m_corr = adam_m[i] / (1 - pow(beta1, t));
        float v_corr = adam_v[i] / (1 - pow(beta2, t));

        params[i] -= lr * m_corr / (sqrt(v_corr) + eps);
    }
    t += 1;
}


SirenNetwork::SirenNetwork(int n_hidden, int hidden_size, int batch_size)
{
    m_batch_size = batch_size;

    m_layers_shapes.push_back(std::pair<int,int>{hidden_size, INPUT_DIM});
    for (int i = 0; i < n_hidden; ++i) {
        m_layers_shapes.push_back(std::pair<int,int>{hidden_size, hidden_size});
    }
    m_layers_shapes.push_back(std::pair<int,int>{OUTPUT_DIM, hidden_size});

    // input will be copied to outputs attr
    int n_params = 0, n_outputs = m_batch_size * INPUT_DIM;

    for (auto [out_dim, in_dim]: m_layers_shapes) {
        n_params += in_dim * out_dim + out_dim;
        // two outputs for linear layer and one for sin activation
        n_outputs += 3 * m_batch_size * out_dim;
    }
    // there is no activation after last linear layer
    n_outputs -= m_batch_size * m_layers_shapes.back().first;

    m_weights_biases = std::vector<float>(n_params);
    m_weights_grads = std::vector<float>(n_params);
    m_outputs = std::vector<float>(n_outputs);
    m_out_grads = std::vector<float>(n_outputs);

    m_adam_m = std::vector<float>(n_params);
    m_adam_v = std::vector<float>(n_params);
    
    m_gt_buffer = std::vector<float>(m_batch_size * OUTPUT_DIM);

    // weights initialization

    std::random_device rd;
    std::mt19937 gen(rd());

    int w_offset = 0;
    for (int i = 0; i < m_layers_shapes.size(); ++i){
        auto [out_dim, in_dim] = m_layers_shapes[i];
    
        float c;
        if (i == 0)
            c = 1.0f / in_dim;
        else
            c = sqrt(6.0f / in_dim) / 30.0f;
        std::uniform_real_distribution<> dis(-c, c);

        for (int i = 0; i < out_dim * in_dim; ++i)
            m_weights_biases[w_offset + i] = dis(gen);
        w_offset += in_dim * out_dim + out_dim;
    }
}


void SirenNetwork::setWeights(const std::vector<float> &weights)
{
    m_weights_biases = weights;
}


std::vector<float> SirenNetwork::getWeights() const
{
    return m_weights_biases;
}


std::vector<float> SirenNetwork::getWeightsGradients() const
{
    return m_weights_grads;
}


std::vector<float> SirenNetwork::getOutputsGradients() const
{
    return m_out_grads;
}


void SirenNetwork::forward(float *res, const float *input, int batch_size)
{
    m_batch_size = batch_size;

    int first_in_dim = m_layers_shapes.front().second;
    for (int i = 0; i < m_batch_size * first_in_dim; ++i) {
        m_outputs[i] = input[i];
    }

    uint32_t w_offset = 0, out_offset = m_batch_size * first_in_dim, in_offset = 0;
    int layer_i = 0;
    for (auto [out_dim, in_dim]: m_layers_shapes) {
        kernel2D_matmul(
            m_outputs.data(), m_weights_biases.data(), m_outputs.data(),
            out_dim, in_dim, m_batch_size,
            out_offset, w_offset, in_offset);
        in_offset = out_offset;
        out_offset += m_batch_size * out_dim;
        w_offset += in_dim * out_dim;

        kernel2D_add_bias(
            m_outputs.data(), m_outputs.data(), m_weights_biases.data(),
            out_dim, m_batch_size,
            out_offset, in_offset, w_offset);
        in_offset = out_offset;
        out_offset += m_batch_size * out_dim;
        w_offset += out_dim;

        if (layer_i < m_layers_shapes.size() - 1) {
            kernel2D_sin_activation(
                m_outputs.data(), m_outputs.data(),
                m_batch_size, out_dim,
                out_offset, in_offset);
            in_offset = out_offset;
            out_offset += m_batch_size * out_dim;
        }

        ++layer_i;
    }

    int last_out_dim = m_layers_shapes.back().first;
    m_outputs_end = out_offset - m_batch_size * last_out_dim;
    for (int i = 0; i < m_batch_size * last_out_dim; ++i) {
        res[i] = m_outputs[m_outputs_end + i];
    }
}


void SirenNetwork::backward(const float *y_gt)
{
    // copy input
    int last_out_dim = m_layers_shapes.back().first;
    for (int i = 0; i < m_batch_size * last_out_dim; ++i) {
        m_gt_buffer[i] = y_gt[i];
    }
    
    // firstly: compute mse gradient
    // shape is [out_dim, batch_size]
    int outputs_offset = m_outputs_end;
    int out_grads_offset = outputs_offset;
    kernel1D_mse_grad(
        m_out_grads.data(), m_outputs.data(), m_gt_buffer.data(),
        m_batch_size,
        out_grads_offset, outputs_offset);
    
    // compute gradients for each layer iteratively
    int w_offset = m_weights_biases.size();
    for (int i = m_layers_shapes.size() - 1; i >= 0; --i) {
        auto [out_dim, in_dim] = m_layers_shapes[i];

        if (i < m_layers_shapes.size() - 1) {
            // compute sin grad
            int out_grad_to_write = out_grads_offset - out_dim * m_batch_size;
            outputs_offset -= m_batch_size * out_dim;
            kernel2D_sin_grad(
                m_out_grads.data(), m_outputs.data(), m_out_grads.data(),
                out_dim, m_batch_size,
                out_grad_to_write, outputs_offset, out_grads_offset);
            out_grads_offset = out_grad_to_write;
        }

        // linear layer gradients: bias, weights, outputs
        outputs_offset -= m_batch_size * out_dim;
        w_offset -= out_dim;
        kernel2D_bias_grad(
            m_weights_grads.data(), m_out_grads.data(),
            out_dim, m_batch_size,
            w_offset, out_grads_offset);

        w_offset -= in_dim * out_dim;
        outputs_offset -= m_batch_size * in_dim;
        kernel2D_matmul_transposed_right(
            m_weights_grads.data(), m_out_grads.data(), m_outputs.data(),
            out_dim, m_batch_size, in_dim,
            w_offset, out_grads_offset, outputs_offset);
        
        int out_grad_to_write = out_grads_offset - in_dim * m_batch_size;
        kernel2D_matmul_transposed_left(
            m_out_grads.data(), m_weights_biases.data(), m_out_grads.data(),
            in_dim, out_dim, m_batch_size,
            out_grad_to_write, w_offset, out_grads_offset);
        out_grads_offset = out_grad_to_write;
    }
}


void SirenNetwork::step(float lr)
{
    kernel1D_Adam_step(
        m_weights_biases.data(), m_weights_grads.data(), m_adam_m.data(), m_adam_v.data(),
        m_weights_biases.size(), lr);
}


std::shared_ptr<SirenNetwork> getSirenNetwork(int n_hidden, int hidden_size, int batch_size)
{
    std::shared_ptr<SirenNetwork> pImpl = nullptr;
    #ifdef USE_VULKAN
    auto ctx = vk_utils::globalContextGet(false, 0);
    pImpl = CreateSirenNetwork_generated(n_hidden, hidden_size, batch_size, ctx, batch_size);
    #else
    pImpl = std::make_shared<SirenNetwork>(n_hidden, hidden_size, batch_size);
    #endif
    return pImpl;
}

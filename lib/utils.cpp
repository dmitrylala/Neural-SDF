#include "utils.h"



std::vector<float> load_floats(const std::string &path)
{
    std::ifstream fin(path, std::ios::binary);
    std::vector<float> values;
    float f;
    while (fin.read(reinterpret_cast<char*>(&f), sizeof(float)))
        values.push_back(f);
    return values;
}


VectorPair load_points(const std::string &sample_path)
{
    std::vector<float> points, gt_sdf;
    std::ifstream fin(sample_path, std::ios::binary);

    int n;
    fin.read(reinterpret_cast<char*>(&n), sizeof(int));

    float f;
    int i = 0;
    while ((i < 3 * n) && fin.read(reinterpret_cast<char*>(&f), sizeof(float))) {
        points.push_back(f);
        ++i;
    }

    while (fin.read(reinterpret_cast<char*>(&f), sizeof(float)))
        gt_sdf.push_back(f);

    return VectorPair{points, gt_sdf};
}


std::vector<int> shuffle_batch_idxs(int n_batches)
{
    std::vector<int> idxs;
    idxs.reserve(n_batches);

    for (int i = 0; i < n_batches; ++i)
        idxs.push_back(i);
 
    std::random_device rd;
    std::mt19937 g(rd());
 
    std::shuffle(idxs.begin(), idxs.end(), g);

    return idxs;
}


std::vector<std::vector<float>> batchify(const std::vector<float> &matrix,
    int batch_size, int n_batches, int n_cols)
{
    std::vector<std::vector<float>> batches;
    
    int n_points = matrix.size() / n_cols;
    for (int i = 0; i < n_batches; ++i) {
        std::vector<float> batch;
        for (int row = i * batch_size; (row < (i + 1) * batch_size) && \
            (row < n_points); ++row) {
            for (int col = 0; col < n_cols; ++col)
                batch.push_back(matrix[row * n_cols + col]);
        }
        batches.push_back(batch);
    }

    return batches;
}


std::vector<float> transpose(const std::vector<float> &m, int n_rows, int n_cols)
{
    std::vector<float> m_transposed(n_cols * n_rows);
    for (int i = 0; i < n_cols; ++i) {
        for (int j = 0; j < n_rows; ++j) {
            m_transposed[i * n_rows + j] = m[j * n_cols + i];
        }
    }
    return m_transposed;
}


float mse_loss(const std::vector<float> &y_pred, const std::vector<float> &y_gt)
{
    float mse = 0.0f;
    for (int i = 0; i < y_pred.size(); ++i) {
        float diff = y_pred[i] - y_gt[i];
        mse += diff * diff;
    }
    return mse / y_pred.size();
}


float3 EyeRayDir(float x, float y, float4x4 a_mViewProjInv)
{
    float4 pos = float4(2.0f * x - 1.0f, 2.0f * y - 1.0f, 0.0f, 1.0f );
    pos = a_mViewProjInv * pos;
    pos /= pos.w;
    return normalize(to_float3(pos));
}


void transform_ray3f(float4x4 a_mWorldViewInv, float3* ray_pos, float3* ray_dir) 
{
    float4 rayPosTransformed = a_mWorldViewInv * to_float4(*ray_pos, 1.0f);
    float4 rayDirTransformed = a_mWorldViewInv * to_float4(*ray_dir, 0.0f);

    (*ray_pos) = to_float3(rayPosTransformed);
    (*ray_dir) = to_float3(normalize(rayDirTransformed));
}


uint32_t RealColorToUint32(float4 real_color)
{
    float r = real_color[0] * 255.0f;
    float g = real_color[1] * 255.0f;
    float b = real_color[2] * 255.0f;
    float a = real_color[3] * 255.0f;

    uint32_t red   = (uint32_t)r;
    uint32_t green = (uint32_t)g;
    uint32_t blue  = (uint32_t)b;
    uint32_t alpha = (uint32_t)a;

    return red | (green << 8) | (blue << 16) | (alpha << 24);
}

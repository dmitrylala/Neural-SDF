#include "read_utils.h"



std::vector<float> read_weights(const std::string &weights_path)
{
    std::ifstream fin(weights_path, std::ios::binary);
    std::vector<float> weights;
    float f;
    while (fin.read(reinterpret_cast<char*>(&f), sizeof(float)))
        weights.push_back(f);
    return weights;
}


VectorPair read_test_points(const std::string &test_points_path)
{
    std::vector<float> points, gt_sdf;
    std::ifstream fin(test_points_path, std::ios::binary);

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

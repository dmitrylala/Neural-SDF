#include "utils.h"



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


std::tuple<int,int,int> parse_network_setup(const ArgParser &parser)
{
    int n_hidden_layers = parser.getOptionValue<int>("--n_hidden");
    int hidden_size = parser.getOptionValue<int>("--hidden_size");
    std::cout << "Architecture params: n_hidden = " << n_hidden_layers << \
        ", hidden_size = " << hidden_size << std::endl;

    int batch_size = parser.getOptionValue<int>("--batch_size");
    std::cout << "Train params: batch_size = " << batch_size << std::endl;

    return std::tuple<int,int,int>{n_hidden_layers, hidden_size, batch_size};
}


std::vector<float> parse_weights(const ArgParser &parser)
{
    std::vector<float> weights;
    bool got_weights_path = parser.hasOption("--weights");
    if (got_weights_path) {
        std::string weights_path = parser.getOptionValue<std::string>("--weights");
        std::cout << "Got weights file: " << weights_path << std::endl;
        weights = read_weights(weights_path);
    }
    return weights;
}


VectorPair parse_test_points(const ArgParser &parser)
{
    bool got_test_points_path = parser.hasOption("--test_points");
    if (got_test_points_path) {
        std::string test_points_path = parser.getOptionValue<std::string>("--test_points");
        std::cout << "Got test points path: " << test_points_path << std::endl;

        auto test_points = read_test_points(test_points_path);

        std::cout << "Loaded points coords: " << test_points.first.size() << \
            ", gt sdfs: " << test_points.second.size() << std::endl;

        return test_points;
    }

    return VectorPair{std::vector<float>(), std::vector<float>()};
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

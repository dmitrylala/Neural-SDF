#include <iostream>
#include <vector>
#include <utility>
#include <fstream>
#include <string>

#include "argparser.h"


using VectorPair = std::pair<std::vector<float>,std::vector<float>>;


std::vector<float> read_weights(const std::string &weights_path);
VectorPair read_test_points(const std::string &test_points_path);

std::vector<float> parse_weights(const ArgParser &parser);
VectorPair parse_test_points(const ArgParser &parser);

std::vector<float> transpose(const std::vector<float> &m, int n_rows, int n_cols);


template <typename T>
void print_first_k(std::ostream &os, const std::vector<T> &vec, int k, std::string sep = " ")
{
    for (int i = 0; i < vec.size() && i < k; ++i)
        os << vec[i] << sep;
    os << std::endl;
}

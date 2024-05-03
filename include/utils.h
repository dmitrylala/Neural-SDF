#include <iostream>
#include <vector>
#include <utility>
#include <fstream>
#include <string>
#include <tuple>

#include "argparser.h"


using VectorPair = std::pair<std::vector<float>,std::vector<float>>;


std::vector<float> read_weights(const std::string &weights_path);
VectorPair read_test_points(const std::string &test_points_path);

std::tuple<int,int,int> parse_network_setup(const ArgParser &parser);
std::vector<float> parse_weights(const ArgParser &parser);
VectorPair parse_test_points(const ArgParser &parser);

std::vector<float> transpose(const std::vector<float> &m, int n_rows, int n_cols);

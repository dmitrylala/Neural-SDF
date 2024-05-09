#pragma once

#include <iostream>
#include <vector>
#include <utility>
#include <fstream>
#include <string>
#include <random>
#include <algorithm>

#include "LiteMath.h"
using namespace LiteMath;


using VectorPair = std::pair<std::vector<float>,std::vector<float>>;


std::vector<float> load_floats(const std::string &weights_path);
VectorPair load_points(const std::string &test_points_path);

std::vector<int> shuffle_batch_idxs(int n_batches);
std::vector<std::vector<float>> batchify(const std::vector<float> &matrix,
    int batch_size, int n_batches, int n_cols);
std::vector<float> transpose(const std::vector<float> &m, int n_rows, int n_cols);

float mse_loss(const std::vector<float> &y_pred, const std::vector<float> &y_gt);

float3 EyeRayDir(float x, float y, float4x4 a_mViewProjInv);
void transform_ray3f(float4x4 a_mWorldViewInv, float3* ray_pos, float3* ray_dir);
uint32_t RealColorToUint32(float4 real_color);


template <typename T>
void print_first_k(std::ostream &os, const std::vector<T> &vec, int k, std::string sep = " ")
{
    for (int i = 0; i < vec.size() && i < k; ++i)
        os << vec[i] << sep;
    os << std::endl;
}

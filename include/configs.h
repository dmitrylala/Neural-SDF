#pragma once


#include "LiteMath.h"
using namespace LiteMath;


struct Camera
{
    float3 pos, look_at, up;
    float field_of_view, z_near, z_far;
};


struct Light
{
    float3 direction;
    float intensity;
};


struct TrainCfg
{
    float lr;
    int n_epochs, log_every_n_epochs;
};


Camera load_cam(const std::string &path);
Light load_light(const std::string &path);
TrainCfg load_train_cfg(const std::string &path);

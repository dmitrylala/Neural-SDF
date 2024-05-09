#include "configs.h"


Camera load_cam(const std::string &path)
{
    float3 pos, look_at, up;
    float fov, z_near, z_far;

    FILE *f = fopen(path.c_str(), "r");

    fscanf(f, "camera_position = %f, %f, %f\n", &pos.x, &pos.y, &pos.z);
    fscanf(f, "target = %f, %f, %f\n", &look_at.x, &look_at.y, &look_at.z);
    fscanf(f, "up = %f, %f, %f\n", &up.x, &up.y, &up.z);
    fscanf(f, "field_of_view  = %f\n", &fov);
    fscanf(f, "z_near  = %f\n", &z_near);
    fscanf(f, "z_far  = %f\n", &z_far);

    fclose(f);

    return Camera({
        .pos = pos,
        .look_at = look_at,
        .up = up,
        .field_of_view = fov,
        .z_near = z_near,
        .z_far = z_far
    });
}


Light load_light(const std::string &path)
{
    float3 direction;
    float intensity;

    FILE *f = fopen(path.c_str(), "r");

    fscanf(f, "light_direction = %f, %f, %f\n", &direction.x, &direction.y, &direction.z);
    fscanf(f, "intensity  = %f\n", &intensity);

    fclose(f);

    return Light({
        .direction = direction,
        .intensity = intensity
    });
}


TrainCfg load_train_cfg(const std::string &path)
{
    float lr;
    int n_epochs, log_every_n_epochs;

    FILE *f = fopen(path.c_str(), "r");

    fscanf(f, "lr = %f\n", &lr);
    fscanf(f, "n_epochs  = %d\n", &n_epochs);
    fscanf(f, "log_every_n_epochs  = %d\n", &log_every_n_epochs);

    fclose(f);

    return TrainCfg({
        .lr = lr,
        .n_epochs = n_epochs,
        .log_every_n_epochs = log_every_n_epochs
    });
}

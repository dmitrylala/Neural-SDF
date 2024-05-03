#!/bin/bash

# cp include/siren.h nn/
bash run_kslicer.sh

# cd example_tracer/shaders_generated/
# bash build.sh && cd ../..

cd nn/shaders_generated/
bash build.sh && cd ../..
# rm siren.h && cd ../
# mv nn/siren_generated.h nn/include/SirenNetwork_generated_ubo.h include/


mkdir -p cmake-build-release && cd cmake-build-release/
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_VULKAN=ON ..
make -j 8 && cd ..

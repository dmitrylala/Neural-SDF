#!/bin/bash

echo "Removing ${BUILD_DIR}"
rm -rf ${BUILD_DIR} CMakeUserPresets.json

echo "Cleaning GPU apps"

for app in "nn" "ray_marcher" ; do
    rm -rf $app/include $app/shaders_generated $app/*generated* ;
done

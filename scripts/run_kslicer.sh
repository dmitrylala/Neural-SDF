#!/bin/bash

if [ ! -f $1 ]; then
  echo "File $1 not found! Run 'setup.sh path/to/kslicer.exe path/to/kslicer/folder' first!"
  exit 1
fi

unset -v kslicer_exe kslicer_directory
for var in kslicer_exe kslicer_directory; do
  IFS= read -r "$var" || break
done < $1

current_directory=$(pwd)
cd "$kslicer_directory" || exit 1


KSLICER_OPTIONS=$(cat <<-END
  -pattern ipv \
  -shaderCC glsl \
  -suffix _generated \
  -I$current_directory/external/LiteMath ignore \
  -stdlibfolder $kslicer_directory/TINYSTL \
  -I$kslicer_directory/TINYSTL ignore \
  -DKERNEL_SLICER \
  -v
END

)


$kslicer_exe $current_directory/ray_marcher/ray_marcher.cpp \
-mainClass "RayMarcher" \
-shaderFolderPrefix "ray_marcher/" \
${KSLICER_OPTIONS}


$kslicer_exe $current_directory/nn/siren.cpp \
-mainClass "SirenNetwork" \
-shaderFolderPrefix "nn/" \
${KSLICER_OPTIONS}


cd $current_directory
for app in "nn" "ray_marcher" ; do
  cd $app/shaders_generated/ && bash build.sh && cd ../.. ;
done

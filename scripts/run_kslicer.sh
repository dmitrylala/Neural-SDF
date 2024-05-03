#!/bin/bash
if [ ! -f $1 ]; then
  echo "File $1 not found! Run 'setup.sh path/to/kslicer.exe path/to/kslicer/folder' first!"
  exit 1
fi
unset -v kslicer_exe kslicer_directory
for var in kslicer_exe kslicer_directory; do
  IFS= read -r "$var" || break
done < $1
current_directory=$(pwd)           # получаем текущую директорию и сохраняем в переменную
cd "$kslicer_directory" || exit 1  # переходим в директорию слайсера, запускать нужно из неё

# $kslicer_exe $current_directory/example_tracer/example_tracer.cpp \
# -mainClass "RayMarcherExample" \
# -pattern "ipv" \
# -shaderCC "glsl" \
# -shaderFolderPrefix "example_tracer/" \
# -suffix "_generated" \
# -I$current_directory/external/LiteMath "ignore" \
# -stdlibfolder "$kslicer_directory/TINYSTL" \
# -I$kslicer_directory/TINYSTL "ignore" \
# -DKERNEL_SLICER \
# -v


$kslicer_exe $current_directory/nn/siren.cpp \
-mainClass "SirenNetwork" \
-pattern "ipv" \
-shaderCC "glsl" \
-shaderFolderPrefix "nn/" \
-suffix "_generated" \
-I$current_directory/external/LiteMath "ignore" \
-stdlibfolder "$kslicer_directory/TINYSTL" \
-I$kslicer_directory/TINYSTL "ignore" \
-DKERNEL_SLICER \
-v

cmake_minimum_required(VERSION 3.16)
project(neural_sdf)

# remove MSVC warnings
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set(CMAKE_CXX_STANDARD 17)
option(USE_VULKAN "Enable GPU implementation via Vulkan" OFF)

find_package(OpenMP)

message(STATUS "Cmake binary dir: " ${CMAKE_BINARY_DIR})
message(STATUS "C++ standard: " ${CMAKE_CXX_STANDARD})

if(USE_VULKAN)

  find_package(Vulkan)

  if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
  elseif(CMAKE_SYSTEM_NAME STREQUAL Linux)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_XLIB_KHR)
  elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_MACOS_MVK)
  endif()

  add_subdirectory(external/volk)

  add_compile_definitions(USE_VOLK)
  add_compile_definitions(USE_VULKAN)

  include_directories(${Vulkan_INCLUDE_DIR})

  include_directories("external/LiteMath"
                      "external/vkutils"
                      "external/volk")

  link_directories("../volk")
else()
  include_directories("external/LiteMath")
endif()

include_directories(".")
if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
endif()

# detect broken memory problems via gcc sanitizers
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
  set(PLATFORM_DEPENDEPNT_LIBS ${Vulkan_LIBRARY} dl)
endif()

set(VULKAN_SOURCES
    ${CMAKE_SOURCE_DIR}/external/vkutils/vk_utils.cpp
    ${CMAKE_SOURCE_DIR}/external/vkutils/vk_copy.cpp
    ${CMAKE_SOURCE_DIR}/external/vkutils/vk_context.cpp
    ${CMAKE_SOURCE_DIR}/external/vkutils/vk_alloc_simple.cpp
    ${CMAKE_SOURCE_DIR}/external/vkutils/vk_buffers.cpp
    ${CMAKE_SOURCE_DIR}/external/vkutils/vk_images.cpp
    ${CMAKE_SOURCE_DIR}/external/vkutils/vk_pipeline.cpp
    ${CMAKE_SOURCE_DIR}/external/vkutils/vk_descriptor_sets.cpp)

set(LITEMATH_SOURCES
    ${CMAKE_SOURCE_DIR}/external/LiteMath/Image2d.cpp)

set(NN_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/nn
    ${CMAKE_SOURCE_DIR}/nn/include
)

add_subdirectory(nn/)
add_subdirectory(lib/)
add_subdirectory(test/unit/)
add_subdirectory(bin/)

project(${PROJECT_NAME}_ray_marcher)


if(USE_VULKAN)
  add_library(${PROJECT_NAME} STATIC
              ray_marcher.cpp
              utils.cpp
              ray_marcher_generated.cpp
              ray_marcher_generated_ds.cpp
              ray_marcher_generated_init.cpp
              ${VULKAN_SOURCES})
else()
  add_library(${PROJECT_NAME} STATIC
              ray_marcher.cpp
              utils.cpp)
endif()

project(${PROJECT_NAME}_ray_marcher)


if(USE_VULKAN)
  add_library(${PROJECT_NAME} STATIC
              ray_marcher.cpp
              ray_marcher_generated.cpp
              ray_marcher_generated_ds.cpp
              ray_marcher_generated_init.cpp
              ${VULKAN_SOURCES})
else()
  add_library(${PROJECT_NAME} STATIC
              ray_marcher.cpp)
endif()


target_link_libraries(${PROJECT_NAME} PUBLIC
                      OpenMP::OpenMP_CXX
                      ${CMAKE_PROJECT_NAME}_nn)

target_include_directories(${PROJECT_NAME} PUBLIC
                          ${NN_INCLUDE_DIRS})

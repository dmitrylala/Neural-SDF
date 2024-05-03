project(${PROJECT_NAME}_nn)


if(USE_VULKAN)
  add_library(${PROJECT_NAME} STATIC
              siren.cpp
              siren_generated.cpp
              siren_generated_ds.cpp
              siren_generated_init.cpp
              ${VULKAN_SOURCES})
else()
  add_library(${PROJECT_NAME} STATIC
              siren.cpp)
endif()

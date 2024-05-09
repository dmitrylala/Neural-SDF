project(${PROJECT_NAME}_lib)

add_library(${PROJECT_NAME} STATIC
            argparser.cpp
            utils.cpp
            ray_marcher.cpp
            configs.cpp
            ${LITEMATH_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
                            ${CMAKE_SOURCE_DIR}/include
                            ${NN_INCLUDE_DIRS})

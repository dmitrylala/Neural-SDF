project(${PROJECT_NAME}_lib)

add_library(${PROJECT_NAME} STATIC
            argparser.cpp
            utils.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC 
                            ${CMAKE_SOURCE_DIR}/include)

project(${PROJECT_NAME}_app)


set(${PROJECT_NAME}_libraries
    ${CMAKE_PROJECT_NAME}_lib
    OpenMP::OpenMP_CXX)

if(USE_VULKAN)
  list(APPEND ${PROJECT_NAME}_libraries
        volk
        "${PLATFORM_DEPENDEPNT_LIBS}")
endif()


add_executable(train
                train.cpp)

target_link_libraries(train LINK_PUBLIC
                      ${CMAKE_PROJECT_NAME}_nn
                      ${${PROJECT_NAME}_libraries})

target_include_directories(train PUBLIC
                            ${CMAKE_SOURCE_DIR}/include
                            ${NN_INCLUDE_DIRS})


add_executable(render
                render.cpp
                ${LITEMATH_SOURCES})

target_link_libraries(render LINK_PUBLIC
                      ${CMAKE_PROJECT_NAME}_ray_marcher
                      ${${PROJECT_NAME}_libraries})

target_include_directories(render PUBLIC
                            ${CMAKE_SOURCE_DIR}/include
                            ${CMAKE_SOURCE_DIR}/ray_marcher
                            ${CMAKE_SOURCE_DIR}/ray_marcher/include)

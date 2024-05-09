project(${PROJECT_NAME}_app)


set(${PROJECT_NAME}_libraries
    ${CMAKE_PROJECT_NAME}_lib
    ${CMAKE_PROJECT_NAME}_nn)

if(USE_VULKAN)
  list(APPEND ${PROJECT_NAME}_libraries
        volk
        "${PLATFORM_DEPENDEPNT_LIBS}")
endif()


add_executable(train
                train.cpp)

target_link_libraries(train LINK_PUBLIC
                      ${${PROJECT_NAME}_libraries})

target_include_directories(train PUBLIC
                            ${CMAKE_SOURCE_DIR}/include
                            ${NN_INCLUDE_DIRS})


add_executable(render
                render.cpp)

target_link_libraries(render LINK_PUBLIC
                      ${${PROJECT_NAME}_libraries})

target_include_directories(render PUBLIC
                            ${CMAKE_SOURCE_DIR}/include
                            ${NN_INCLUDE_DIRS})

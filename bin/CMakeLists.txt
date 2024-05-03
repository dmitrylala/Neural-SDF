project(${PROJECT_NAME}_app)


add_executable(train
                train.cpp)


set(${PROJECT_NAME}_libraries
    ${CMAKE_PROJECT_NAME}_lib
    ${CMAKE_PROJECT_NAME}_nn
    OpenMP::OpenMP_CXX)

if(USE_VULKAN)
  list(APPEND ${PROJECT_NAME}_libraries
        volk
        "${PLATFORM_DEPENDEPNT_LIBS}")
endif()

target_link_libraries(train LINK_PUBLIC ${${PROJECT_NAME}_libraries})

target_include_directories(train PUBLIC
                            ${CMAKE_SOURCE_DIR}/include
                            ${CMAKE_SOURCE_DIR}/nn
                            ${CMAKE_SOURCE_DIR}/nn/include)

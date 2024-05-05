project(${PROJECT_NAME}_unit_tests)

find_package(Catch2 REQUIRED)

set(EXE_SOURCES
	siren.cpp
)

add_executable(nn_test ${EXE_SOURCES})

target_link_libraries(nn_test PRIVATE
                        ${CMAKE_PROJECT_NAME}_nn
                        Catch2::Catch2WithMain)

target_include_directories(nn_test PRIVATE
                            ${NN_INCLUDE_DIRS})

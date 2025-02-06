cmake_minimum_required(VERSION 3.5)
project(test)
include_directories(${PROJECT_SOURCE_DIR}/include; /usr/local/include/opencv4)
aux_source_directory(${PROJECT_SOURCE_DIR}/src Srcs)
set(CMAKE_CXX_STANDARD 17)

link_directories(/usr/local/lib)
set(LIBRARY_DIR /usr/local/lib)
file(GLOB LIBRARY_FILES "${LIBRARY_DIR}/*.so")

add_executable(test main.cpp; ${Srcs})

target_link_libraries(test ${LIBRARY_FILES})
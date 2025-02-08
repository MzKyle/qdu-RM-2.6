cmake_minimum_required(VERSION 2.5)
 
project(recognize CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)

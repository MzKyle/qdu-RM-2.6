cmake_minimum_required(VERSION 3.5)

# 设置项目名称
project(TEST)

# 设置C++标准，这里使用C++17
set(CMAKE_CXX_STANDARD 17)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 添加头文件搜索路径
include_directories()

# 添加可执行文件
add_executable(test try_cpp_detcter_armor.cpp)

# 链接OpenCV库
target_link_libraries(test ${OpenCV_LIBS})
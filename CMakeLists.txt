cmake_minimum_required(VERSION 3.21)
cmake_policy(VERSION 3.21.3...3.27)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)



project(kimage LANGUAGES HIP)

add_executable(main main.hip)

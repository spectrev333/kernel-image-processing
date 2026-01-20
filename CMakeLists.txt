cmake_minimum_required(VERSION 3.21)
cmake_policy(VERSION 3.21.3...3.27)
project(kimage LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(stb)

if(NOT TARGET stb)
  add_library(stb INTERFACE)
  target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})
endif()

add_definitions(-D__HIP_PLATFORM_AMD__=1)
set(CMAKE_MODULE_PATH "/opt/rocm/lib/cmake/hip" ${CMAKE_MODULE_PATH})
find_package(HIP)

include_directories("/opt/rocm/include")

hip_add_executable(main main.cpp conv_cpu.cpp conv_gpu.hip)
#target_include_directories(main PRIVATE /opt/rocm/include)
target_link_libraries(main PRIVATE stb)
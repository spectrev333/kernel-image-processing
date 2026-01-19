cmake_minimum_required(VERSION 3.21)
cmake_policy(VERSION 3.21.3...3.27)
project(kimage LANGUAGES HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)


FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        master # O un commit hash specifico per stabilità
)
FetchContent_MakeAvailable(stb)

if(NOT TARGET stb)
  add_library(stb INTERFACE)
  target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})
endif()


add_executable(main main.hip)
target_link_libraries(main PRIVATE stb)

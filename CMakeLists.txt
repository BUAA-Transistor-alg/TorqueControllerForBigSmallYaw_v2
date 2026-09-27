cmake_minimum_required(VERSION 3.16)

project(TorqueControllerForBigSmallYaw_v2
    VERSION 1.0.0
    DESCRIPTION "Two-link (big/small yaw) dynamics simulator with prescribed base motion"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Python 通过 ctypes 加载该动态库，必须生成位置无关代码。
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# 仿真核心 + C 语言接口
#
# 说明：dynamic_model/ 下的参考代码不参与编译，仅作动力学模型参考；
#       本工程在 src/include 中重新实现了该模型。
# ---------------------------------------------------------------------------
add_library(tcbss SHARED
    src/dynamics.cpp
    src/simulator.cpp
    src/capi.cpp
)

target_include_directories(tcbss
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(tcbss PRIVATE -Wall -Wextra -Wpedantic)

# 产物统一放到 build/ 根目录，方便 python/ 下按相对路径加载。
set_target_properties(tcbss PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "tcbss 动态库输出目录: ${CMAKE_BINARY_DIR}")

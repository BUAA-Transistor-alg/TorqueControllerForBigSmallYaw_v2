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
# 目录约定
#   include/dm + src/dm : 动力学模型核心，C++ 内容位于命名空间 tcbs::dm
#   include/capi + src/capi : C 语言接口（extern "C"，导出 tcbs_* 符号）
#
# 说明：dynamic_model/ 下的参考代码不参与编译，仅作动力学模型参考；
#       本工程在 src/include 中重新实现了该模型。
# ---------------------------------------------------------------------------
set(DM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include/dm)
set(CAPI_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include/capi)

add_library(tcbs SHARED
    src/dm/dynamics.cpp
    src/dm/simulator.cpp
    src/dm/trajectory.cpp
    src/dm/param_gradient.cpp
    src/capi/capi_dm.cpp
)

target_include_directories(tcbs
    PUBLIC ${DM_INCLUDE} ${CAPI_INCLUDE}
)

target_compile_options(tcbs PRIVATE -Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------------------
# 轨迹梯度模块是计算热区（每步 4 子步 × 4 阶段），单独提高优化等级。
# 用生成器表达式按编译器选择，避免逐个探测优化选项。
# ---------------------------------------------------------------------------
set_source_files_properties(
    src/dm/trajectory.cpp src/dm/param_gradient.cpp PROPERTIES
    COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3;-funroll-loops;-fno-math-errno>")

# 产物统一放到 build/ 根目录，方便 python/ 下按相对路径加载。
set_target_properties(tcbs PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "tcbs 动态库输出目录: ${CMAKE_BINARY_DIR}")

# ---------------------------------------------------------------------------
# 自检程序：轨迹梯度（离散伴随）与参数梯度（前向灵敏度）vs 中心差分
# 运行: ./build/tcbs_trajectory_test
# ---------------------------------------------------------------------------
add_executable(tcbs_trajectory_test
    tests/test_trajectory_gradient.cpp
    src/dm/dynamics.cpp
    src/dm/trajectory.cpp
    src/dm/param_gradient.cpp
)
target_include_directories(tcbs_trajectory_test PRIVATE ${DM_INCLUDE})
target_compile_options(tcbs_trajectory_test PRIVATE -Wall -Wextra -Wpedantic)
set_source_files_properties(tests/test_trajectory_gradient.cpp PROPERTIES
    COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3>")
set_target_properties(tcbs_trajectory_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

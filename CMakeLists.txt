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
#   include/com + src/com : 上位机通信（串口协议 + 编码器映射），命名空间 tcbs::com
#   include/capi + src/capi : C 语言接口（extern "C"，导出 tcbs_* 符号）
#   include/mpc + src/mpc : 力矩 MPC（Ceres 梯度法 + dm 解析力矩导数），命名空间 tcbs::mpc
#   include/common + src/common : 通用工具（帧率统计），命名空间 tcbs
#   include/RobotController.hpp + src/RobotController.cpp : 一体化控制封装，命名空间 tcbs
#
# 说明：dynamic_model/ 下的参考代码不参与编译，仅作动力学模型参考；
#       本工程在 src/include 中重新实现了该模型。
#
# 依赖：com 的 SerialProtocol 用 libudev 读 USB iProduct 以筛选串口（需 libudev），
#       并以 std::thread 做收发/重连线程（需 Threads）；
#       mpc 用 Ceres 的 FirstOrderFunction + GradientProblemSolver 做无约束梯度优化
#       （梯度来自 dm::simulateAndGradient 的解析离散伴随，不用自动求导，需 Ceres）。
# ---------------------------------------------------------------------------
set(DM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include/dm)
set(COM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include/com)
set(CAPI_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include/capi)
set(MPC_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include/mpc)
set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include/common)
# 顶层 include/：RobotController.hpp 直接放在总文件夹，供外部以 "RobotController.hpp" 引入
set(TCBS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Threads REQUIRED)
find_package(Ceres REQUIRED)
find_library(TCBS_UDEV_LIBRARY udev)
if(NOT TCBS_UDEV_LIBRARY)
    message(FATAL_ERROR "libudev not found（com/SerialProtocol.hpp 依赖它筛选串口）")
endif()

add_library(tcbs SHARED
    src/dm/dynamics.cpp
    src/dm/simulator.cpp
    src/dm/trajectory.cpp
    src/dm/param_gradient.cpp
    src/dm/param_gradient_batch.cpp
    src/mpc/mpc_controller.cpp
    src/mpc/dual_yaw_mpc_controller.cpp
    src/mpc/mcu_mpc_controller.cpp
    src/common/FrameRateCounter.cpp
    src/RobotController.cpp
    src/capi/capi_dm.cpp
    src/capi/capi_com.cpp
    src/com/CRC.cpp
    src/com/McuDataPreprocessor.cpp
    src/com/FullStrictPoseBuilder.cpp
)

target_include_directories(tcbs
    PUBLIC ${DM_INCLUDE} ${COM_INCLUDE} ${CAPI_INCLUDE} ${MPC_INCLUDE} ${COMMON_INCLUDE}
           ${TCBS_INCLUDE_DIR}
)

target_link_libraries(tcbs PUBLIC ${TCBS_UDEV_LIBRARY} Threads::Threads Ceres::ceres)

target_compile_options(tcbs PRIVATE -Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------------------
# 轨迹梯度模块是计算热区（每步 4 子步 × 4 阶段），单独提高优化等级。
# 用生成器表达式按编译器选择，避免逐个探测优化选项。
# ---------------------------------------------------------------------------
set_source_files_properties(
    src/dm/trajectory.cpp src/dm/param_gradient.cpp PROPERTIES
    COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3;-funroll-loops;-fno-math-errno>")

# ---------------------------------------------------------------------------
# 批量参数梯度（param_gradient_batch）额外开启 SIMD pragma；可选 -march=native。
#
# 只作用于这一个翻译单元：其他模式（trajectory 力矩梯度 / 单序列参数梯度 /
# Simulator）的编译选项与数值行为完全不变。
#
# TCBS_NATIVE_ARCH=ON（默认）时用 -march=native，向量宽度跟随本机（AVX2/AVX512）；
# 需要在别的机器上运行同一个 .so 时用 -DTCBS_NATIVE_ARCH=OFF 重新配置。
# ---------------------------------------------------------------------------
option(TCBS_NATIVE_ARCH "为批量参数梯度开启 -march=native（本机编译本机运行）" ON)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" TCBS_HAS_MARCH_NATIVE)
set(TCBS_PGB_FLAGS "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3;-funroll-loops;-fno-math-errno;-fopenmp-simd>")
if(TCBS_NATIVE_ARCH AND TCBS_HAS_MARCH_NATIVE)
    list(APPEND TCBS_PGB_FLAGS "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native>")
    message(STATUS "批量参数梯度: 开启 -march=native")
else()
    message(STATUS "批量参数梯度: 不开启 -march=native（可移植构建）")
endif()
set_source_files_properties(src/dm/param_gradient_batch.cpp PROPERTIES
    COMPILE_OPTIONS "${TCBS_PGB_FLAGS}")

# 产物统一放到 build/ 根目录，方便 python/ 下按相对路径加载。
set_target_properties(tcbs PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "tcbs 动态库输出目录: ${CMAKE_BINARY_DIR}")

# TorqueControllerForBigSmallYaw_v2

二自由度平面双连杆（big / small yaw）系统的动力学仿真器：

- C++ 实现「动力学模型 + RK4 积分」，支持**细化倍数**（每个 `dt` 内的 RK4 子步数）；
- 提供 **C 语言接口**（`include/tcbss_capi.h`），编译为 `build/libtcbss.so`；
- `python/` 下用 **ctypes** 包装该接口，并用 **pygame** 做实时二阶摆演示。

## 目录结构

```
.
├── build.sh                     # 既有构建脚本：在 build/ 中 cmake + make
├── CMakeLists.txt               # 编译 src/ 与 include/ 为 libtcbss.so
├── dynamic_model/
│   └── deepseek_cpp_...cpp      # 动力学模型参考实现（已修正，见下文）
├── include/
│   ├── params.hpp               # 系统参数（无默认值，默认构造被 delete）
│   ├── state.hpp                # 广义坐标状态 / 状态导数
│   ├── dynamics.hpp             # 动力学模型声明
│   ├── rk4.hpp                  # 通用 RK4 单步积分器（模板，与模型解耦）
│   ├── simulator.hpp            # Simulator：细化倍数、状态设置、step()
│   └── tcbss_capi.h             # C 语言接口
├── src/
│   ├── dynamics.cpp             # 动力学模型实现
│   ├── simulator.cpp            # 仿真器实现（RK4 子步 + theta_c 外推）
│   └── capi.cpp                 # C 接口实现
├── python/
│   ├── tcbss.py                 # ctypes 包装（Params / State / Simulator）
│   └── pendulum_sim.py          # pygame 实时仿真演示（参数写死为常量）
└── test/
    └── verify_dynamics.cpp      # 离线验证程序（不参与默认构建）
```

## 编译

```bash
./build.sh              # 等价于在 build/ 中 cmake .. && make -j
# 产物：build/libtcbss.so
```

`dynamic_model/` 下的参考代码**不参与编译**，仅作模型参考。

## 运行 pygame 演示

```bash
python3 python/pendulum_sim.py
```

所有参数（物理参数、`dt`、细化倍数、按键力矩、初值、`theta_c`）都写死在
[`python/pendulum_sim.py`](python/pendulum_sim.py) 顶部的常量区，**不接受命令行参数**。

| 按键 | 作用 |
| --- | --- |
| `A` / `S` | 关节 b 力矩 `+TORQUE_B` / `-TORQUE_B`（等大反向） |
| `K` / `L` | 关节 s 力矩 `+TORQUE_S` / `-TORQUE_S`（等大反向） |
| `SPACE` | 暂停 / 继续 |
| `R` | 复位到初始状态 |
| `ESC` | 退出 |

坐标系：向右为 `x` 正方向，向上为 `y` 正方向（屏幕 y 轴已翻转）。
绘制约定：连杆 b 由原点画到关节 `R(psi_b)·(Dx,Dy)`，连杆 s 由关节画到
`R(psi_s)·(2Ps)`。

## C++ 接口

```cpp
#include "simulator.hpp"

tcbss::Params params(
    /*mb*/ 1.5, /*Ib*/ 0.030, /*Pbx*/ 0.18, /*Pby*/ 0.0,
    /*ms*/ 0.4, /*Is*/ 0.020, /*Psx*/ 0.10, /*Psy*/ 0.0,
    /*Dx*/ 0.30, /*Dy*/ 0.0, /*gx*/ 0.0, /*gy*/ -9.81,
    /*fbc*/ 0.02, /*fbv*/ 0.05, /*fsc*/ 0.01, /*fsv*/ 0.02, /*lambda*/ 100.0);

tcbss::Simulator sim(params, /*dt*/ 1e-3, /*refinement*/ 4);

// 单独设置当前两个广义坐标的位置与速度
sim.setState(theta_b, dtheta_b, theta_s, dtheta_s);
sim.setThetaB(theta_b, dtheta_b);
sim.setThetaS(theta_s, dtheta_s);

// 输入两个驱动力矩与 theta_c 的位置/速度/加速度，演化 dt
tcbss::State next = sim.step(Tb, Ts, theta_c, dtheta_c, ddtheta_c);
```

要点：

- `Params` **没有默认值**（`Params() = delete`），17 个参数必须全部显式给出；
- `Simulator(params, dt, refinement)` 同样无默认参数，`dt > 0`、`refinement >= 1`，
  否则抛 `std::invalid_argument`；
- `refinement` 即细化倍数：把 `dt` 均分为 `refinement` 个 RK4 子步；
- `step()` 的 RK4 子步内，`theta_c` 按**等角加速度外推**：
  `theta_c(τ) = theta_c + dtheta_c·τ + ½·ddtheta_c·τ²`，`τ ∈ [0, dt]`；
  本步结束后的 `theta_c` 由调用方自行按同一公式推进。

## C 语言接口

见 [`include/tcbss_capi.h`](include/tcbss_capi.h)：不透明句柄 `TcbssSimulator*`，
`tcbss_create` / `tcbss_destroy` / `tcbss_set_state` / `tcbss_set_theta_b` /
`tcbss_set_theta_s` / `tcbss_step` / `tcbss_get_*`，失败信息用 `tcbss_last_error()` 获取。

## Python 接口

```python
from tcbss import Params, State, Simulator

with Simulator(params, dt=1e-3, refinement=4) as sim:
    sim.set_state(State(theta_b=-1.5708, dtheta_b=0.0, theta_s=0.0, dtheta_s=0.0))
    st = sim.step(Tb=0.0, Ts=0.0, theta_c=0.0, dtheta_c=0.0, ddtheta_c=0.0)
```

`Params` / `State` 为 frozen dataclass，字段全部必填。动态库路径可用环境变量
`TCBSS_LIB` 覆盖，默认查找 `build/libtcbss.so`。

## 动力学模型修正说明

`dynamic_model/` 中最初的参考实现与正向运动学不自洽，具体为：

1. 质量矩阵 `M11`、`M12` 漏掉了连杆 s 绕关节 s 的惯量 `ms·|Ps|² + Is`；
2. 重力项 `G1` 漏掉了连杆 s 的重力贡献（即 `G2`，它通过 `psi_b` 的旋转同样作用于 `theta_b`）；
3. 科氏项 `C1` 漏掉了一项 `ms·(dh/dtheta_s)·dtheta_s·dpsi_b`。

症状：重力项不是任何势函数的梯度（`∂G1/∂θs = 0` 而 `∂G2/∂θb ≠ 0`），
且零力矩零摩擦下能量自增 —— 从偏离竖直 0.15 rad 静止释放，20 s 内角速度被泵到
**10.52 rad/s**；修正后同样初值只有 **2.88 rad/s**，20 s 能量漂移 `2.2e-14`。

现已在 `src/dynamics.cpp` 与 `dynamic_model/` 的参考实现中同步修正为：

```
M11 = mb|Pb|² + Ib + ms|D|² + (ms|Ps|² + Is) + 2·ms·h
M12 =                         (ms|Ps|² + Is) +     ms·h
M22 =                         (ms|Ps|² + Is)

C1 = ms·(dh/dtheta_s)·dtheta_s·(2·dpsi_b + dtheta_s)
C2 = -ms·(dh/dtheta_s)·dpsi_b²

G1 = [连杆 b 项] + G2        （关节 s 随连杆 b 转动，故 s 的重力也进入 G1）
G2 = ms(gx·Psx + gy·Psy)·sin(psi_s) + ms(gx·Psy − gy·Psx)·cos(psi_s)

F1 = Qb − C1 − G1 − M11·ddtheta_c
F2 = Qs − C2 − G2 − M12·ddtheta_c     （基座加速度的耦合列就是质量矩阵第一列）
```

其中 `h = Dx·Psx·cos(θs) + Dy·Psy·cos(θs) + Dy·Psx·sin(θs) − Dx·Psy·sin(θs)`。
原始参考实现保存在 git 提交 `74d3074` 中。

## 独立验证

[`test/verify_dynamics.cpp`](test/verify_dynamics.cpp) 用一个**完全独立的数值 oracle**
（正向运动学 Jacobian → 3x3 质量矩阵、势函数数值梯度、数值 Christoffel 符号）
对拍解析实现，并检查积分精度与能量守恒：

```bash
g++ -std=c++17 -O2 -I include test/verify_dynamics.cpp src/dynamics.cpp src/simulator.cpp \
    -o build/tcbss_verify && ./build/tcbss_verify
```

最近一次结果：

| 检查 | 结果 |
| --- | --- |
| 与修正后参考实现逐点对拍（30 万组随机输入） | 最大误差 `0.0` |
| 质量矩阵（含 `theta_c` 耦合列）解析 vs 数值 | `1.2e-10` |
| 重力项解析 vs 数值势梯度 | `1.7e-09` |
| 科氏项解析 vs 数值 Christoffel | `3.4e-10` |
| 端到端加速度（含 `theta_c` 运动）vs oracle | `1.4e-08` |
| 细化倍数收敛性（`dt=0.01`，refine 1→32） | `2.6e-04` → `6.3e-10`（约 h⁴） |
| 单大步(细化 400) vs 250 小步 | `6.5e-14` |
| 零力矩零摩擦自由演化 20 s 能量漂移 | `2.2e-14` |

// tools/control_demo.cpp — 使用 RobotController 一体化封装的双级 yaw 正弦演示
//
// 移植自 /home/huhu233/rm2027/TorqueController/src/control_demo.cpp（tcs 单轴版）。
//
// ── 与原版的差异 ──────────────────────────────────────────────────────────
//   * 命名空间 tcs → tcbs；包含路径 "tcs/RobotController.h" → "RobotController.hpp"。
//   * 被控对象是**双连杆**（大 yaw + 小 yaw）:
//       原版 set(yaw, pitch) 里的 yaw 是**关节角**；本工程 RobotController::set 的
//       两个 yaw 目标是**世界系方位角** psi_b / psi_s（rad），内部再换算成关节角下发。
//       两轴目标各自自动 wrap 到与当前世界方位角差最小的等效角。
//   * 构造函数形态不同: 原版把 J / tau_c / b / Q / R / Rd / max_iter 等一串标量直接
//       传给 RobotController；本工程改为传**结构化配置**：
//         dm::Params（双连杆动力学）/ MPCController::Options（求解器）/ 
//         DualYawMpcController::Options（两轴积分增益），另需 imu_location。
//   * 就绪判据: 原版等 fused.valid（融合滤波器）；本工程没有 FusionFilter，
//       改为等 MCU 与 IMU 都收到过有效样本。
//   * 本工程 MPC 用预 tanh 增量 + 解析伴随梯度，**必须**给足 RK4 子步
//       （Options::refinement，见 include/mpc/mpc_controller.hpp 的稳定性说明），
//       因此取 python/sim_config.py 的 REFINEMENT = 16，而不是原版的 1 个子步。
//
// ── 演示轨迹（3s 周期正弦，两轴相位差 0、pitch 相位差 90°）──
//   psi_b(t) = +30°·sin(ωt)                      大 yaw 世界方位角往复摆动
//   psi_s(t) = psi_b(t)                          ⇒ θ_s ≡ 0，小 yaw 始终跟随大 yaw 回中
//   pitch(t) = (5° + 15°·sin(ωt − 90°))          −10° ~ +20° 关节角
//
// ── 运行 ──
//   ./build/tcbs_control_demo              # 默认：不使用重力（模型无重力项）
//   ./build/tcbs_control_demo --gravity    # 使用重力：实测 gx/gy 每步送进 MPC
//   Ctrl+C 退出

#include "RobotController.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <thread>

namespace tcbs {
namespace {

std::atomic<bool> g_running{true};
void signalHandler(int) { g_running = false; }

constexpr double DEG2RAD = M_PI / 180.0;
constexpr double PERIOD_S = 3.0;              // 正弦周期（秒）
constexpr double OMEGA = 2.0 * M_PI / PERIOD_S;

// ── 控制周期与 MPC 配置 ──
constexpr double DT_CTRL     = 0.01;          // 控制周期 (100 Hz)
constexpr int    MPC_PRED_N  = 20;            // int(DELAY_TIME / DT_CTRL)，DELAY_TIME=0.2s
constexpr int    MPC_REFINE  = 16;            // 每主步 RK4 子步数（见文件头说明）
constexpr int    MAX_ITER    = 30;
constexpr double MAX_TORQUE_B = 1.0;          // 大 yaw 软限幅 [N·m]
constexpr double MAX_TORQUE_S = 1.0;          // 小 yaw 软限幅 [N·m]

// ── MPC 权重（语义取自原 control_demo 的 Q / R / Rd）──
//   原版代价 = Q·sqrt(err²+a) + R·mean(τ²) + Rd·mean(Δτ²)（单轴）
//   本工程代价（见 include/mpc/mpc_controller.hpp）= Σ w_psi·mean(psi 误差)
//              + w_dpsi·mean(dpsi 误差) + w_tau·mean(τ²)
//              + w_x_b·mean(x_b²) + w_x_s·mean(x_s²)
//              + w_dx_b·mean(Δx_b²) + w_dx_s·mean(Δx_s²)
//   （内层六项与两个外层 L2 都按大 / 小 yaw 各一份权重；每个 mean 都是**该通道
//     自己** N 步的均值）
//   因此位置上 Q→w_psi、力矩幅值上 R→w_tau、增量上 Rd→w_dx_b / w_dx_s；
//   w_x_b / w_x_s 是"预 tanh 量"的 L2 惩罚，原版没有对应项，但本工程要求它们 > 0
//   （把 x 约束在 tanh 线性区、保住梯度），故取与 R 同量级的小正数。
constexpr double W_PSI_B  = 1.0;              // 大 yaw ← 原 Q
constexpr double W_PSI_S  = 1.0;              // 小 yaw ← 原 Q
constexpr double W_DPSI_B = 0.0;              // 大 yaw
constexpr double W_DPSI_S = 0.0;              // 小 yaw
constexpr double W_TAU_B  = 0.0;             // 大 yaw ← 原 R
constexpr double W_TAU_S  = 0.0;             // 小 yaw ← 原 R

constexpr double W_X_B    = 1.0;             // 大 yaw：原版无此项；必须 > 0
constexpr double W_X_S    = 0.1;             // 小 yaw：同上
constexpr double W_DX_B   = 10.0;              // 大 yaw ← 原 Rd
constexpr double W_DX_S   = 0.1;              // 小 yaw ← 原 Rd
constexpr double INTEGRAL_GAIN_B = 0.0;      // 大 yaw 积分补偿比例系数
constexpr double INTEGRAL_GAIN_S = 0.0;      // 小 yaw 积分补偿比例系数

// ===========================================================================
//   gx / gy 是"摆平面内的等效重力分量"，运行期由 FullStrictPoseBuilder 反解得到。
//   **是否真的进模型**由 MPCController::Options::use_gravity 决定（本文件里经
//   makeMpcOptions(use_gravity) 设置，**默认 true = 使用重力**，
//   命令行 --no-gravity 可切成对照模式，见 main）：
//     * 默认（不加命令行参数）= 不使用重力 ⇒ 模型里没有重力项，setGravity 收到的
//       实测值会被求解器按 0 存，重力矩完全靠反馈扛（做对照实验用）；
//     * 加 --gravity / --use-gravity 才让实测 gx/gy 真正参与求解。
//   因此这里的初值只在通信尚未就绪时有影响，底盘水平时可填 0；
//   实测值仍然照常从 rc.getState().strict 读出，不受开关影响。
//   lambda 是摩擦平滑常数，是**已知固定模型常数**而非辨识量。
// ===========================================================================
dm::Params makeDynamicsParams() {
    return dm::Params(
        /*mb=*/    1.8271,    /*Ib=*/  0.00237991,  /*Pbx=*/ 0.0408972,  /*Pby=*/ 0.0225917,
        /*ms=*/    0.331926,   /*Is=*/  0.000344412,  /*Psx=*/ -0.000204337,  /*Psy=*/ -5.95902e-05,
        /*Dx=*/    -0.299491,   /*Dy=*/  -0.184257,
        /*gx=*/    0.0,    /*gy=*/  0.0,
        /*fbc=*/   0.00226296,  /*fbv=*/ 0.0993675,
        /*fsc=*/   0.000403778,  /*fsv=*/ 0.00145031,
        /*lambda=*/100.0, /*kb=*/ 4.0, /*ks=*/ 0.045313);
}

mpc::MPCController::Options makeMpcOptions(bool use_gravity) {
    mpc::MPCController::Options opt;
    opt.dt           = DT_CTRL;
    opt.refinement   = MPC_REFINE;
    opt.N            = MPC_PRED_N;
    opt.max_torque_b = MAX_TORQUE_B;
    opt.max_torque_s = MAX_TORQUE_S;
    opt.w_psi_b      = W_PSI_B;
    opt.w_psi_s      = W_PSI_S;
    opt.w_dpsi_b     = W_DPSI_B;
    opt.w_dpsi_s     = W_DPSI_S;
    opt.w_tau_b      = W_TAU_B;
    opt.w_tau_s      = W_TAU_S;
    opt.w_x_b        = W_X_B;
    opt.w_x_s        = W_X_S;
    opt.w_dx_b       = W_DX_B;
    opt.w_dx_s       = W_DX_S;
    opt.max_iter     = MAX_ITER;
    opt.use_gravity  = use_gravity;   // true（默认）= 实测 gx/gy 进模型
    return opt;
}

mpc::DualYawMpcController::Options makeWrapperOptions() {
    mpc::DualYawMpcController::Options opt;
    opt.integral_gain_b = INTEGRAL_GAIN_B;
    opt.integral_gain_s = INTEGRAL_GAIN_S;
    return opt;
}

// ── 正弦目标（世界系方位角，rad）──
inline double targetPsiB(double t) {
    return 30.0 * DEG2RAD * std::sin(OMEGA * t);
}

// 小 yaw 世界方位角 = 大 yaw 世界方位角 ⇒ 小 yaw 关节角 θ_s ≡ 0（跟随回中）
inline double targetPsiS(double t) {
    return targetPsiB(t);
}

// pitch 关节角目标: 范围 -10°~+20°（中点 5°、幅值 15°），与 yaw 相位差 90°
inline double targetPitch(double t) {
    return (5.0 + 15.0 * std::sin(OMEGA * t - M_PI / 2.0)) * DEG2RAD;
}

}  // namespace
}  // namespace tcbs

// main() 必须留在全局命名空间（否则不是程序入口）；
// 下面把 namespace tcbs 内的名字引入作用域，便于 main() 直接使用。
using namespace tcbs;

int main(int argc, char** argv) {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // 重力开关：直接写进 MPCController::Options::use_gravity（不再单独传参）。
    // **默认使用重力**（实测 gx/gy 每步真正进 MPC 模型）：
    //   --no-gravity 显式关闭（模型里没有重力项，做对照实验）
    //   --gravity / --use-gravity 冗余打开（便于脚本里写清楚）
    bool use_gravity = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gravity") == 0
            || std::strcmp(argv[i], "--use-gravity") == 0) {
            use_gravity = true;
        } else if (std::strcmp(argv[i], "--no-gravity") == 0) {
            use_gravity = false;
        } else {
            printf("未知参数: %s（可用: --no-gravity / --gravity / --use-gravity）\n",
                   argv[i]);
            return 2;
        }
    }

    printf("=== tcbs_control_demo (RobotController, 双级 yaw) ===\n");
    printf("正弦周期 %.1fs: psi_b ±30°, psi_s = psi_b (θ_s≡0), pitch -10°~+20°, 相位差 90°\n",
           PERIOD_S);
    printf("重力: %s（%s）\n",
           use_gravity ? "使用" : "不使用",
           use_gravity ? "实测 gx/gy 每步送进 MPC 模型"
                       : "use_gravity=false，求解器内部恒按 (0,0) 处理");

    // 一体化控制封装：通信 + 严格反解 + 双级 yaw MPC + 后台发送线程
    // imu_location:     IMU 安装构型（决定严格反解的运动学链）
    // mpc_loop_period:  McuMpcController 后台 loop 周期（秒，必须传参；0.01 = 100Hz）
    // mcu_linear_params 用当前默认标定值（LinearParams{} 与默认构造等价）
    RobotController rc(com::FullStrictPoseBuilder::ImuLocation::ON_HEAD,
                       makeDynamicsParams(),
                       makeMpcOptions(use_gravity),
                       makeWrapperOptions(),
                       /*mpc_loop_period=*/0.01,
                       com::McuDataPreprocessor::LinearParams{},
                       /*sequence_mode=*/false);

    // 等待 MCU 与 IMU 都收到有效数据
    // （本工程无 FusionFilter，"就绪"等价于两个来源都已有过样本）
    printf("等待 MCU 与 IMU 数据就绪...\n");
    while (g_running) {
        auto st = rc.getState();
        if (st.mcu.valid && st.imu.valid) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!g_running) return 0;
    printf("数据就绪，开始正弦控制（Ctrl+C 退出）\n");

    auto t0 = std::chrono::steady_clock::now();
    int loop = 0;
    while (g_running) {
        auto start = std::chrono::steady_clock::now();
        double t = std::chrono::duration<double>(start - t0).count();

        // 设置发送参数 + MPC 目标（后台线程 100Hz 求解并发送）
        // 两轴都用"仅力矩"模式（YawMode 1）：电控只施加 MPC 力矩，不叠加内环
        double psi_b = targetPsiB(t);
        double psi_s = targetPsiS(t);
        double pitch = targetPitch(t);
        rc.set(/*auto_aim_enable=*/true, /*yaw_torque_only_mode=*/true,
               psi_b, psi_s, pitch, /*fire=*/false, /*integral_enable=*/false);

        // 每 0.1s 打印一次
        if (++loop % 10 == 0) {
            auto st = rc.getState();
            printf("[t=%6.2fs] tgt psi_b=%+7.2f° psi_s=%+7.2f° pitch=%+6.2f° | "
                   "meas psi_b=%+7.2f° psi_s=%+7.2f° | "
                   "th_b=%+.3f th_s=%+.3f | tau_b=%+.4f tau_s=%+.4f | "
                   "valid=%d loop_fps=%.1f | T_big=%d T_small=%d\n",
                   t,
                   psi_b / DEG2RAD, psi_s / DEG2RAD, pitch / DEG2RAD,
                   st.strict.big_azimuth / DEG2RAD,
                   st.strict.small_azimuth / DEG2RAD,
                   st.strict.yaw_big_angle, st.strict.yaw_small_angle,
                   st.mpc.yaw_big_torque, st.mpc.yaw_small_torque,
                   (int)(st.mcu.valid && st.imu.valid), st.mpc.loop_fps,
                   (int)st.mcu.yaw_big_temperature, (int)st.mcu.yaw_small_temperature);
        }

        // 循环结束处等待到 start + 10ms（100Hz），不累计误差
        std::this_thread::sleep_until(start + std::chrono::milliseconds(10));
    }

    printf("\n退出。\n");
    return 0;
}

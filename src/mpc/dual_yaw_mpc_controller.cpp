#include "dual_yaw_mpc_controller.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "angle_wrap.hpp"   // 每拍把参考整圈对齐到实测世界方位角（见 solveWith 的 2b 段）

namespace tcbs {
namespace mpc {

DualYawMpcController::DualYawMpcController(com::RobotCommunication* comm,
                                           const dm::Params& params,
                                           const MPCController::Options& mpc_options,
                                           const Options& options)
    : comm_(comm),
      n_(mpc_options.N),
      // mpc_ 的构造会完成全部配置校验（dt / N / refinement / 限幅 / 权重）。
      mpc_(params, mpc_options),
      options_(options) {
    if (!(options_.integral_gain_b >= 0.0) || !(options_.integral_gain_s >= 0.0)) {
        throw std::invalid_argument("DualYawMpcController: integral_gain must be >= 0");
    }
}

void DualYawMpcController::reset() {
    target_buf_b_.clear();
    target_buf_s_.clear();
    integral_b_ = 0.0;
    integral_s_ = 0.0;
    prev_pred_psi_b_ = 0.0;
    prev_pred_psi_s_ = 0.0;
    has_prev_pred_ = false;
    mpc_.reset();
}

void DualYawMpcController::fillBuffer(std::vector<double>& buf,
                                      const std::vector<double>& seq,
                                      int n) {
    buf.clear();
    const std::size_t count =
        std::min<std::size_t>(seq.size(), static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < count; ++i) {
        buf.push_back(seq[i]);
    }
    // 不足 N 个时用最后一个值补齐；序列为空则补 0。
    const double last = (count > 0) ? buf.back() : 0.0;
    while (static_cast<int>(buf.size()) < n) {
        buf.push_back(last);
    }
}

DualYawMpcController::Result DualYawMpcController::step(double target_psi_b,
                                                        double target_psi_s,
                                                        bool integral_enable_b,
                                                        bool integral_enable_s) {
    // 延迟目标缓冲（最多 N 个，最旧在前）。
    target_buf_b_.push_back(target_psi_b);
    target_buf_s_.push_back(target_psi_s);
    while (static_cast<int>(target_buf_b_.size()) > n_) {
        target_buf_b_.erase(target_buf_b_.begin());
    }
    while (static_cast<int>(target_buf_s_.size()) > n_) {
        target_buf_s_.erase(target_buf_s_.begin());
    }
    return solve(integral_enable_b, integral_enable_s);
}

DualYawMpcController::Result DualYawMpcController::step(
    const std::vector<double>& target_psi_b_buf,
    const std::vector<double>& target_psi_s_buf,
    bool integral_enable_b,
    bool integral_enable_s) {
    fillBuffer(target_buf_b_, target_psi_b_buf, n_);
    fillBuffer(target_buf_s_, target_psi_s_buf, n_);
    return solve(integral_enable_b, integral_enable_s);
}

DualYawMpcController::Measurement DualYawMpcController::measure() const {
    Measurement m;
    if (comm_ == nullptr) {
        return m;
    }
    // 严格反解包 → 两个广义坐标 + 基座量；世界方位角按 dm 模型定义合成，
    // 与内层 loss 的 psi_b / psi_s 语义保持一致。
    const com::FullStrictPoseBuilder::StrictPose pose = comm_->getStrictPose();

    m.valid = true;
    m.theta_c = pose.chassis_azimuth;
    m.dtheta_c = pose.chassis_omega;
    m.theta_b = pose.yaw_big_angle;
    m.dtheta_b = pose.big_motor_omega;
    m.theta_s = pose.yaw_small_angle;
    m.dtheta_s = pose.small_motor_omega;

    m.psi_b = m.theta_c + m.theta_b;
    m.psi_s = m.psi_b + m.theta_s;
    m.dpsi_b = m.dtheta_c + m.dtheta_b;
    m.dpsi_s = m.dpsi_b + m.dtheta_s;
    // 旋转平面内重力分量：底盘俯仰/横滚变化时随样本更新，solve 会送进 MPC。
    m.gx = pose.gx;
    m.gy = pose.gy;
    return m;
}

DualYawMpcController::Result DualYawMpcController::solve(bool integral_enable_b,
                                                         bool integral_enable_s) {
    Result r;

    // ---- 1. 读严格反解包，按模型定义组装状态、世界方位角与重力 ----
    const Measurement meas = measure();
    if (!meas.valid) {
        return r;
    }

    // 延迟缓冲被清空时（reset 后直接求解）退化为 N 个 0 目标。
    if (target_buf_b_.empty()) target_buf_b_.assign(1, 0.0);
    if (target_buf_s_.empty()) target_buf_s_.assign(1, 0.0);

    // ---- 2. 参考序列：世界系目标直接使用 ----
    // trajectory 内部已按 theta_c / dtheta_c / ddtheta_c 外推基座，
    // 这里**不再**叠加 (i+1)·dt·ω_c 之类的底盘补偿。
    std::vector<double> ref_b = target_buf_b_;
    std::vector<double> ref_s = target_buf_s_;
    while (static_cast<int>(ref_b.size()) < n_) ref_b.push_back(ref_b.back());
    while (static_cast<int>(ref_s.size()) < n_) ref_s.push_back(ref_s.back());

    return solveWith(meas, ref_b, ref_s, target_buf_b_.front(), target_buf_s_.front(),
                     integral_enable_b, integral_enable_s);
}

DualYawMpcController::Result DualYawMpcController::step(
    const Measurement& measurement,
    const std::vector<double>& ref_psi_b,
    const std::vector<double>& ref_psi_s,
    bool integral_enable_b,
    bool integral_enable_s) {
    Result r;
    if (!measurement.valid) {
        return r;
    }

    // 空序列按目标 0；不足 N 时用最后一个值补齐。
    std::vector<double> ref_b = ref_psi_b.empty() ? std::vector<double>(1, 0.0) : ref_psi_b;
    std::vector<double> ref_s = ref_psi_s.empty() ? std::vector<double>(1, 0.0) : ref_psi_s;
    while (static_cast<int>(ref_b.size()) < n_) ref_b.push_back(ref_b.back());
    while (static_cast<int>(ref_s.size()) < n_) ref_s.push_back(ref_s.back());

    return solveWith(measurement, ref_b, ref_s, ref_b.front(), ref_s.front(),
                     integral_enable_b, integral_enable_s);
}

DualYawMpcController::Result DualYawMpcController::solveWith(
    const Measurement& measurement,
    const std::vector<double>& ref_b,
    const std::vector<double>& ref_s,
    double target_psi_b,
    double target_psi_s,
    bool integral_enable_b,
    bool integral_enable_s) {
    Result r;

    const double theta_c = measurement.theta_c;
    const double dtheta_c = measurement.dtheta_c;
    const double psi_b = measurement.psi_b;
    const double psi_s = measurement.psi_s;

    r.state_psi_b = psi_b;
    r.state_psi_s = psi_s;
    r.state_theta_b = measurement.theta_b;
    r.state_theta_s = measurement.theta_s;
    r.state_theta_c = theta_c;
    r.state_dtheta_c = dtheta_c;

    // ---- 2b. ★ **每拍**（100Hz）把参考对齐到当拍实测世界方位角所在圈 ----
    //   只做「整体平移 2π 的整数倍」：
    //     · 序列形状（斜率/曲率/各点相对关系）完全不变，跟踪误差也原样保留；
    //     · 实测与参考本来就相差 < π 时是恒等变换（正常运行路径零副作用）；
    //     · 只在「圈」不一致时（底盘方位角跨过 ±π、参考来自不同圈等）把整条参考拉回
    //       同一圈——否则代价里会出现 w_psi_b·(2π)² 的整圈残差，MPC 会打出恒定幅值的
    //       力矩脉冲（每圈一次）。根因修复见 FullStrictPoseBuilder::unwrapChassisAzimuth，
    //       这里是**兜底**：任何环节的整圈表示差都不再能变成控制量脉冲。
    //   ★ 放在 solveWith() 而不是 McuMpcController::set()：本函数是标量目标、
    //     序列目标、显式测量三条调用路径的唯一汇聚点，且每拍必到（set() 只有每帧一次）。
    std::vector<double> ref_b_aligned = ref_b;
    std::vector<double> ref_s_aligned = ref_s;
    alignSeqToNearest(ref_b_aligned, psi_b);
    alignSeqToNearest(ref_s_aligned, psi_s);
    const double target_b_aligned = wrapToNearest(target_psi_b, psi_b);
    const double target_s_aligned = wrapToNearest(target_psi_s, psi_s);

    r.target_psi_b = target_b_aligned;
    r.target_psi_s = target_s_aligned;

    MPCController::Reference reference;
    reference.psi_b = ref_b_aligned;
    reference.psi_s = ref_s_aligned;

    dm::State x0;
    x0.theta_b = measurement.theta_b;
    x0.dtheta_b = measurement.dtheta_b;
    x0.theta_s = measurement.theta_s;
    x0.dtheta_s = measurement.dtheta_s;

    // ---- 3. 重力更新：底盘俯仰/横滚变化 ⇒ 旋转平面内重力分量变化 ----
    // 底盘水平时 gx = gy = 0（重力全在 z 轴、平面内无分量），与 dm 模型一致。
    // 照常传实测值；MPCController::Options::use_gravity = false（对照模式）时
    // 求解器内部会把它按 (0,0) 存，模型里就没有重力项。
    mpc_.setGravity(measurement.gx, measurement.gy);

    // ---- 4. MPC 求解（返回两轴第一步力矩与预测序列，不发送）----
    const MPCController::Result mres =
        mpc_.step(x0, theta_c, dtheta_c, base_ddtheta_c_, reference);

    r.valid = mres.ok;
    r.torque_b = mres.torque_b;
    r.torque_s = mres.torque_s;
    r.pred_psi_b = mres.pred_psi_b;
    r.pred_psi_s = mres.pred_psi_s;
    r.pred_dpsi_b = mres.pred_dpsi_b;
    r.pred_dpsi_s = mres.pred_dpsi_s;
    // 世界系 → 关节系（下发 MCU 的目标角/角速度用的就是这两个）
    r.pred_theta_b = r.pred_psi_b - theta_c;
    r.pred_theta_s = r.pred_psi_s - r.pred_psi_b;
    r.pred_dtheta_b = r.pred_dpsi_b - dtheta_c;
    r.pred_dtheta_s = r.pred_dpsi_s - r.pred_dpsi_b;
    r.ref_psi_b = ref_b_aligned;
    r.ref_psi_s = ref_s_aligned;
    r.pred_psi_b_seq = mres.pred_psi_b_seq;
    r.pred_psi_s_seq = mres.pred_psi_s_seq;

    // ---- 5. 积分补偿（两轴各自独立：累积量、增益、开关都分开）----
    // 第一次 step 无上一步预测，不计算积分增量。
    if (integral_enable_b) {
        if (has_prev_pred_) {
            integral_b_ += options_.integral_gain_b * (prev_pred_psi_b_ - psi_b);
        }
        r.torque_b = std::clamp(r.torque_b + integral_b_, -mpc_.maxTorqueB(),
                                mpc_.maxTorqueB());
    } else {
        // 大 yaw 未启用：只清空大 yaw 的积分值，力矩保持 MPC 结果。
        integral_b_ = 0.0;
    }
    if (integral_enable_s) {
        if (has_prev_pred_) {
            integral_s_ += options_.integral_gain_s * (prev_pred_psi_s_ - psi_s);
        }
        r.torque_s = std::clamp(r.torque_s + integral_s_, -mpc_.maxTorqueS(),
                                mpc_.maxTorqueS());
    } else {
        // 小 yaw 未启用：只清空小 yaw 的积分值，力矩保持 MPC 结果。
        integral_s_ = 0.0;
    }
    // 两轴的"上一步预测"都照常记录（下一次 step 算积分增量要用），与开关无关：
    // 某一轴关掉再打开时不会拿到过期的预测值。
    prev_pred_psi_b_ = r.pred_psi_b;
    prev_pred_psi_s_ = r.pred_psi_s;
    has_prev_pred_ = true;
    r.integral_b = integral_b_;
    r.integral_s = integral_s_;

    return r;
}

}  // namespace mpc
}  // namespace tcbs

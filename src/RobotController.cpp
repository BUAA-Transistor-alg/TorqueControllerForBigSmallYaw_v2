#include "RobotController.hpp"

namespace tcbs {

RobotController::RobotController(
    com::FullStrictPoseBuilder::ImuLocation imu_location,
    const dm::Params& params,
    const mpc::MPCController::Options& mpc_options,
    const mpc::DualYawMpcController::Options& wrapper_options,
    double mpc_loop_period,
    const com::McuDataPreprocessor::LinearParams& mcu_linear_params,
    bool sequence_mode)
    : sequence_mode_(sequence_mode),
      comm_(imu_location, mcu_linear_params),
      mcu_mpc_(&comm_, params, mpc_options, wrapper_options, mpc_loop_period) {
    mcu_mpc_.start();  // 启动后台发送线程（周期 = mpc_loop_period）
}

RobotController::~RobotController() {
    // mcu_mpc_ 析构自动 stop + join 后台线程；comm_ 析构停止串口线程。
    // 成员按声明逆序析构 ⇒ mcu_mpc_ 先于 comm_，顺序正确。
}

RobotController::State RobotController::getState() {
    State st;

    // ── MCU / IMU 原始数据（同一次加锁的一致快照）──
    auto raw = comm_.getLatestData();
    if (raw.mcu_valid) {
        const auto& m = raw.mcu_packet;
        st.mcu.valid              = true;
        st.mcu.bullet_velocity    = m.bullet_velocity;
        st.mcu.pitch_angle        = m.pitch_angle;
        st.mcu.yaw_big_angle      = m.yaw_big_angle;
        st.mcu.yaw_big_omega      = m.yaw_big_omega;
        st.mcu.yaw_small_angle    = m.yaw_small_angle;
        st.mcu.yaw_small_omega    = m.yaw_small_omega;
        st.mcu.chassis_imu_yaw    = m.chassis_imu_yaw;
        st.mcu.chassis_imu_omega  = m.chassis_imu_omega;
        st.mcu.mark               = m.mark;
        st.mcu.color              = m.color;
        st.mcu.auto_aim_switch    = m.auto_aim_switch;
        st.mcu.yaw_big_temperature   = m.yaw_big_temperature;
        st.mcu.yaw_small_temperature = m.yaw_small_temperature;
        st.mcu.mcu2_seq           = raw.mcu2_seq;
    }
    if (raw.imu_valid) {
        const auto& im = raw.imu_packet;
        st.imu.valid = true;
        st.imu.gx = im.gx;
        st.imu.gy = im.gy;
        st.imu.gz = im.gz;
        st.imu.ax = im.ax;
        st.imu.ay = im.ay;
        st.imu.az = im.az;
        st.imu.euler_yaw   = im.euler_yaw;
        st.imu.euler_pitch = im.euler_pitch;
        st.imu.euler_roll  = im.euler_roll;
        st.imu.dt_one_tenth_ms = im.dt_one_tenth_ms;
    }

    // ── 严格反解数据包（独立输出，始终可读，无 valid 标志）──
    st.strict = comm_.getStrictPose();

    // ── MPC 状态（含参考/预测序列与积分值）──
    const mpc::McuMpcController::State m = mcu_mpc_.state();
    st.mpc.yaw_big_target_angle    = m.yaw_big_target_angle;
    st.mpc.yaw_big_target_velocity = m.yaw_big_target_velocity;
    st.mpc.yaw_big_torque          = m.yaw_big_torque;
    st.mpc.yaw_small_target_angle    = m.yaw_small_target_angle;
    st.mpc.yaw_small_target_velocity = m.yaw_small_target_velocity;
    st.mpc.yaw_small_torque          = m.yaw_small_torque;
    st.mpc.pred_psi_b          = m.pred_psi_b;
    st.mpc.pred_psi_s          = m.pred_psi_s;
    st.mpc.delayed_target_b    = m.delayed_target_b;
    st.mpc.delayed_target_s    = m.delayed_target_s;
    st.mpc.set_target_b        = m.set_target_b;
    st.mpc.set_target_s        = m.set_target_s;
    st.mpc.integral_b          = m.integral_b;
    st.mpc.integral_s          = m.integral_s;
    st.mpc.loop_fps            = m.loop_fps;
    st.mpc.ticks_since_set     = m.ticks_since_set;
    st.mpc.ref_psi_b_seq       = m.ref_psi_b_seq;
    st.mpc.ref_psi_s_seq       = m.ref_psi_s_seq;
    st.mpc.pred_psi_b_seq      = m.pred_psi_b_seq;
    st.mpc.pred_psi_s_seq      = m.pred_psi_s_seq;

    return st;
}

void RobotController::set(bool auto_aim_enable, bool yaw_torque_only_mode_b,
                          bool yaw_torque_only_mode_s, double target_psi_b,
                          double target_psi_s, double pitch_target_angle, bool fire,
                          bool integral_enable_b, bool integral_enable_s) {
    if (sequence_mode_) {
        throw std::runtime_error("RobotController: SEQUENCE mode selected, "
                                 "use sequence set() instead of single set()");
    }
    mcu_mpc_.set(auto_aim_enable, yaw_torque_only_mode_b, yaw_torque_only_mode_s,
                 target_psi_b, target_psi_s, pitch_target_angle, fire,
                 integral_enable_b, integral_enable_s);
}

void RobotController::set(bool auto_aim_enable, bool yaw_torque_only_mode_b,
                          bool yaw_torque_only_mode_s,
                          const std::vector<double>& target_psi_b_seq,
                          const std::vector<double>& target_psi_s_seq,
                          const std::vector<double>& pitch_seq,
                          const std::vector<bool>& fire_seq,
                          bool integral_enable_b, bool integral_enable_s) {
    if (!sequence_mode_) {
        throw std::runtime_error("RobotController: SINGLE mode selected, "
                                 "use single set() instead of sequence set()");
    }
    mcu_mpc_.set(auto_aim_enable, yaw_torque_only_mode_b, yaw_torque_only_mode_s,
                 target_psi_b_seq, target_psi_s_seq, pitch_seq, fire_seq,
                 integral_enable_b, integral_enable_s);
}

}  // namespace tcbs

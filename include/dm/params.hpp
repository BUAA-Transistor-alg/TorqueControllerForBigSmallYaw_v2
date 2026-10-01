#pragma once

// 双连杆系统的物理、重力与摩擦参数。
//
// 本结构体刻意不提供任何默认值：默认构造被删除，只能通过全参数构造函数
// 创建，从而在编译期杜绝“少传参数”。

namespace tcbs {
namespace dm {

struct Params {
    // ---- 连杆 b ----
    double mb;      ///< 质量 [kg]
    double Ib;      ///< 绕质心的转动惯量 [kg*m^2]
    double Pbx;     ///< 质心在连杆 b 局部坐标下的 x [m]
    double Pby;     ///< 质心在连杆 b 局部坐标下的 y [m]

    // ---- 连杆 s ----
    double ms;      ///< 质量 [kg]
    double Is;      ///< 绕质心的转动惯量 [kg*m^2]
    double Psx;     ///< 质心在连杆 s 局部坐标下的 x [m]
    double Psy;     ///< 质心在连杆 s 局部坐标下的 y [m]

    // ---- 关节偏移与重力场 ----
    double Dx;      ///< 关节 s 相对关节 b 的偏移量 x [m]
    double Dy;      ///< 关节 s 相对关节 b 的偏移量 y [m]
    double gx;      ///< 重力场加速度 x [m/s^2]
    double gy;      ///< 重力场加速度 y [m/s^2]

    // ---- 摩擦 ----
    double fbc;     ///< 关节 b 库仑摩擦系数
    double fbv;     ///< 关节 b 粘滞摩擦系数
    double fsc;     ///< 关节 s 库仑摩擦系数
    double fsv;     ///< 关节 s 粘滞摩擦系数
    double lambda;  ///< 平滑摩擦力参数（通常取 100）

    // ---- 控制力矩通道增益（"下发值 → 实际电机力矩"的比例）----
    //
    // ★ 本工程里 **输入到模型的 Tb / Ts 永远是"发给电控的指令值"**（协议规定该值
    //   恒在 [-1, +1]，电控/电机内部再换算成力矩）。真实作用到连杆上的物理力矩是
    //        κ_b = kb·Tb,   κ_s = ks·Ts
    //   所以这两个 k 是**动力学模型的一部分**，必须和物理参数一起辨识/给定。
    //
    //   实测口径（Sentry1）：kb = 4 表示"下发 1 ⇒ 电机实际发力 4 N·m"。
    //   ks 未实测，取值只决定参数的数值标度（见 identify_params 的说明）：
    //   模型只约束乘积 kb·(惯量项)，绝对尺度需要一个外部锚。
    //
    //   注意符号：大 yaw 电机反向安装带来的 −1 已经由
    //   McuDataPreprocessor::send_big_torque_scale（以及数据集里的记录值）处理完，
    //   这里的 kb 只表示幅值比例，取 +4。若以后实测发现记录口径里还差一个符号，
    //   再把 kb 取负即可（本结构允许负值）。
    double kb;      ///< 关节 b 控制力矩增益 [N·m / 指令单位]
    double ks;      ///< 关节 s 控制力矩增益 [N·m / 指令单位]

    Params() = delete;

    /// 前 17 个参数之后**追加** kb / ks（带默认值 1.0）。
    ///
    /// 为什么给默认值：本结构有 30+ 个构造点（含 tools/ 与 build/mpcchk/ 下的
    /// 校验程序），全部改一遍噪声大；而 kb = ks = 1 恰好是"旧行为"（指令值即物理
    /// 力矩），因此默认值只在"没有专门配过增益"的旧场景里生效，不会静默改变已有
    /// 校验程序的语义。**凡是描述真实机器人/真实数据的路径都必须显式给出**。
    Params(double mb_,
           double Ib_,
           double Pbx_,
           double Pby_,
           double ms_,
           double Is_,
           double Psx_,
           double Psy_,
           double Dx_,
           double Dy_,
           double gx_,
           double gy_,
           double fbc_,
           double fbv_,
           double fsc_,
           double fsv_,
           double lambda_,
           double kb_ = 1.0,
           double ks_ = 1.0)
        : mb(mb_),
          Ib(Ib_),
          Pbx(Pbx_),
          Pby(Pby_),
          ms(ms_),
          Is(Is_),
          Psx(Psx_),
          Psy(Psy_),
          Dx(Dx_),
          Dy(Dy_),
          gx(gx_),
          gy(gy_),
          fbc(fbc_),
          fbv(fbv_),
          fsc(fsc_),
          fsv(fsv_),
          lambda(lambda_),
          kb(kb_),
          ks(ks_) {}
};

}  // namespace dm
}  // namespace tcbs

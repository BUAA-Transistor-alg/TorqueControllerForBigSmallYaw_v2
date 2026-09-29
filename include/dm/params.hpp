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

    Params() = delete;

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
           double lambda_)
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
          lambda(lambda_) {}
};

}  // namespace dm
}  // namespace tcbs

#ifndef TCBSS_CAPI_H
#define TCBSS_CAPI_H

#include <stddef.h>

/*
 * TorqueControllerForBigSmallYaw_v2 仿真器的 C 语言接口。
 *
 * 供 ctypes / C / 其它语言调用。所有角度单位为 rad，力矩为 N*m，时间为 s。
 *
 * 生命周期：
 *     TcbssSimulator* sim = tcbss_create(&params, dt, refinement);
 *     if (!sim) { fprintf(stderr, "%s\n", tcbss_last_error()); }
 *     ...
 *     tcbss_destroy(sim);
 *
 * 本文件不使用任何 C++ 特性，可直接被 C 编译器包含。
 */

#ifdef __cplusplus
extern "C" {
#endif

/* 不透明句柄：内部是 C++ 的 tcbss::Simulator。 */
typedef struct TcbssSimulator TcbssSimulator;

/* 不透明句柄：内部是 C++ 的有限多步轨迹梯度求解器（离散伴随法）。 */
typedef struct TcbssTrajectory TcbssTrajectory;

/* 系统参数。字段顺序与 tcbss::Params 完全一致。 */
typedef struct TcbssParams {
    double mb;      /* 连杆 b 质量 */
    double Ib;      /* 连杆 b 转动惯量 */
    double Pbx;     /* 连杆 b 质心局部坐标 x */
    double Pby;     /* 连杆 b 质心局部坐标 y */

    double ms;      /* 连杆 s 质量 */
    double Is;      /* 连杆 s 转动惯量 */
    double Psx;     /* 连杆 s 质心局部坐标 x */
    double Psy;     /* 连杆 s 质心局部坐标 y */

    double Dx;      /* 关节偏移量 x */
    double Dy;      /* 关节偏移量 y */
    double gx;      /* 重力场加速度 x */
    double gy;      /* 重力场加速度 y */

    double fbc;     /* 关节 b 库仑摩擦系数 */
    double fbv;     /* 关节 b 粘滞摩擦系数 */
    double fsc;     /* 关节 s 库仑摩擦系数 */
    double fsv;     /* 关节 s 粘滞摩擦系数 */
    double lambda_; /* 平滑摩擦力参数（通常取 100） */
} TcbssParams;

/* 两个广义坐标的位置与速度。 */
typedef struct TcbssState {
    double theta_b;
    double dtheta_b;
    double theta_s;
    double dtheta_s;
} TcbssState;

/* ------------------------------------------------------------------ */
/* 构造 / 析构                                                        */
/* ------------------------------------------------------------------ */

/*
 * 创建仿真器。params 不可为 NULL，dt 必须 > 0，refinement 必须 >= 1。
 * 成功返回句柄；失败返回 NULL，可通过 tcbss_last_error() 获取原因。
 * 必须用 tcbss_destroy() 释放。
 */
TcbssSimulator* tcbss_create(const TcbssParams* params, double dt, int refinement);

/* 销毁仿真器；传入 NULL 是安全的。 */
void tcbss_destroy(TcbssSimulator* sim);

/* 最近一次失败的错误信息（线程局部），无错误时返回空字符串。 */
const char* tcbss_last_error(void);

/* ------------------------------------------------------------------ */
/* 状态设置 / 读取                                                    */
/* ------------------------------------------------------------------ */

/* 直接设置当前两个广义坐标的位置与速度。 */
void tcbss_set_state(TcbssSimulator* sim, TcbssState state);
void tcbss_set_generalized(TcbssSimulator* sim,
                           double theta_b,
                           double dtheta_b,
                           double theta_s,
                           double dtheta_s);

/* 分别设置单个广义坐标的位置与速度。 */
void tcbss_set_theta_b(TcbssSimulator* sim, double theta_b, double dtheta_b);
void tcbss_set_theta_s(TcbssSimulator* sim, double theta_s, double dtheta_s);

/* 读取当前状态；out 不可为 NULL。 */
void tcbss_get_state(const TcbssSimulator* sim, TcbssState* out);

/* ------------------------------------------------------------------ */
/* 参数查询                                                           */
/* ------------------------------------------------------------------ */

double tcbss_get_dt(const TcbssSimulator* sim);
int tcbss_get_refinement(const TcbssSimulator* sim);
void tcbss_get_params(const TcbssSimulator* sim, TcbssParams* out);

/* ------------------------------------------------------------------ */
/* 仿真一步                                                           */
/* ------------------------------------------------------------------ */

/*
 * 输入两个驱动力矩、theta_c 的位置/速度/加速度，推进一个 dt。
 * out 可为 NULL（此时只更新内部状态）。
 */
void tcbss_step(TcbssSimulator* sim,
                double Tb,
                double Ts,
                double theta_c,
                double dtheta_c,
                double ddtheta_c,
                TcbssState* out);

/* ==================================================================== */
/* 有限多步轨迹 + 损失对每一步力矩的解析梯度（离散伴随法）              */
/* ==================================================================== */

/*
 * 损失定义（时间均值，k = 0 .. K-1，共六项）：
 *
 *   L = (1/2K) * sum_k [ w_psi_b      * (psi_b[k]      - target_psi_b[k])^2
 *                      + w_psi_s      * (psi_s[k]      - target_psi_s[k])^2
 *                      + w_dpsi_b     * (dpsi_b[k]     - target_dpsi_b[k])^2
 *                      + w_dpsi_s     * (dpsi_s[k]     - target_dpsi_s[k])^2
 *                      + w_tau_b      * tau_b[k]^2
 *                      + w_tau_s      * tau_s[k]^2 ]
 *
 *   psi_b  = theta_c + theta_b          psi_s  = theta_c + theta_b + theta_s
 *   dpsi_b = dtheta_c + dtheta_b        dpsi_s = dtheta_c + dtheta_b + dtheta_s
 *
 * 目标序列传 NULL 表示该项目标恒为 0（退化为惩罚幅值）。
 * theta_c 序列为外部给定、不参与优化的已知量，每步内按等角加速度外推。
 */

/* 轨迹求解器：创建一次，缓冲可跨多次调用复用（内部按最大 K 自动扩容）。 */
TcbssTrajectory* tcbss_trajectory_create(const TcbssParams* params);

/* 销毁；传入 NULL 是安全的。 */
void tcbss_trajectory_destroy(TcbssTrajectory* t);

/*
 * 仅前向：计算损失值，并可导出四个全局量序列。
 *
 *   theta_c0    本序列起始时刻的基座角度
 *   dtheta_c    base 角速度（全序列常值，除非 ddtheta_c != 0）
 *   ddtheta_c   base 角加速度（每步内常值）
 *   dt          每步时间 [s]
 *   num_steps   K
 *   tau         NULL 时全部步使用 (tau_b_fixed, tau_s_fixed)；否则长度 2K
 *   out_*       长度 K 的输出，可为 NULL
 *
 * 失败返回 -1 并通过 tcbss_last_error() 给出原因。
 */
double tcbss_trajectory_loss(const TcbssTrajectory* t,
                             double theta_c0,
                             double dtheta_c,
                             double ddtheta_c,
                             double dt,
                             size_t num_steps,
                             const double* tau,
                             double tau_b_fixed,
                             double tau_s_fixed,
                             const TcbssState* x0,
                             double w_psi_b,
                             double w_psi_s,
                             double w_dpsi_b,
                             double w_dpsi_s,
                             double w_tau_b,
                             double w_tau_s,
                             const double* target_psi_b,
                             const double* target_psi_s,
                             const double* target_dpsi_b,
                             const double* target_dpsi_s,
                             double* out_psi_b,
                             double* out_psi_s,
                             double* out_dpsi_b,
                             double* out_dpsi_s);

/*
 * 正向 + 反向伴随：返回损失值，并写出 dL/dtau（长度 2K，dL/dtau[2k] = dL/dTb_k）。
 * grad_tau 不可为 NULL。out_final_state 可为 NULL。
 * 失败返回 -1 并通过 tcbss_last_error() 给出原因。
 */
double tcbss_trajectory_gradient(TcbssTrajectory* t,
                                 double theta_c0,
                                 double dtheta_c,
                                 double ddtheta_c,
                                 double dt,
                                 size_t num_steps,
                                 const double* tau,
                                 const TcbssState* x0,
                                 double w_psi_b,
                                 double w_psi_s,
                                 double w_dpsi_b,
                                 double w_dpsi_s,
                                 double w_tau_b,
                                 double w_tau_s,
                                 const double* target_psi_b,
                                 const double* target_psi_s,
                                 const double* target_dpsi_b,
                                 const double* target_dpsi_s,
                                 double* grad_tau,
                                 TcbssState* out_final_state);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* TCBSS_CAPI_H */

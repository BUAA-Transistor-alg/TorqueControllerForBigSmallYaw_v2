#ifndef CAPI_DM_H
#define CAPI_DM_H

#include <stddef.h>

/*
 * TorqueControllerForBigSmallYaw_v2 仿真器的 C 语言接口。
 *
 * 供 ctypes / C / 其它语言调用。所有角度单位为 rad，力矩为 N*m，时间为 s。
 *
 * 生命周期：
 *     TcbsSimulator* sim = tcbs_create(&params, dt, refinement);
 *     if (!sim) { fprintf(stderr, "%s\n", tcbs_last_error()); }
 *     ...
 *     tcbs_destroy(sim);
 *
 * 本文件不使用任何 C++ 特性，可直接被 C 编译器包含。
 */

#ifdef __cplusplus
extern "C" {
#endif

/* 不透明句柄：内部是 C++ 的 tcbs::Simulator。 */
typedef struct TcbsSimulator TcbsSimulator;

/* 不透明句柄：内部是 C++ 的有限多步轨迹梯度求解器（离散伴随法）。 */
typedef struct TcbsTrajectory TcbsTrajectory;

/* 系统参数。字段顺序与 tcbs::Params 完全一致。 */
typedef struct TcbsParams {
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
} TcbsParams;

/* 两个广义坐标的位置与速度。 */
typedef struct TcbsState {
    double theta_b;
    double dtheta_b;
    double theta_s;
    double dtheta_s;
} TcbsState;

/* ------------------------------------------------------------------ */
/* 构造 / 析构                                                        */
/* ------------------------------------------------------------------ */

/*
 * 创建仿真器。params 不可为 NULL，dt 必须 > 0，refinement 必须 >= 1。
 * 成功返回句柄；失败返回 NULL，可通过 tcbs_last_error() 获取原因。
 * 必须用 tcbs_destroy() 释放。
 */
TcbsSimulator* tcbs_create(const TcbsParams* params, double dt, int refinement);

/* 销毁仿真器；传入 NULL 是安全的。 */
void tcbs_destroy(TcbsSimulator* sim);

/* 最近一次失败的错误信息（线程局部），无错误时返回空字符串。 */
const char* tcbs_last_error(void);

/* ------------------------------------------------------------------ */
/* 状态设置 / 读取                                                    */
/* ------------------------------------------------------------------ */

/* 直接设置当前两个广义坐标的位置与速度。 */
void tcbs_set_state(TcbsSimulator* sim, TcbsState state);
void tcbs_set_generalized(TcbsSimulator* sim,
                           double theta_b,
                           double dtheta_b,
                           double theta_s,
                           double dtheta_s);

/* 分别设置单个广义坐标的位置与速度。 */
void tcbs_set_theta_b(TcbsSimulator* sim, double theta_b, double dtheta_b);
void tcbs_set_theta_s(TcbsSimulator* sim, double theta_s, double dtheta_s);

/* 读取当前状态；out 不可为 NULL。 */
void tcbs_get_state(const TcbsSimulator* sim, TcbsState* out);

/* ------------------------------------------------------------------ */
/* 参数查询                                                           */
/* ------------------------------------------------------------------ */

double tcbs_get_dt(const TcbsSimulator* sim);
int tcbs_get_refinement(const TcbsSimulator* sim);
void tcbs_get_params(const TcbsSimulator* sim, TcbsParams* out);

/* ------------------------------------------------------------------ */
/* 仿真一步                                                           */
/* ------------------------------------------------------------------ */

/*
 * 输入两个驱动力矩、theta_c 的位置/速度/加速度，推进一个 dt。
 * out 可为 NULL（此时只更新内部状态）。
 */
void tcbs_step(TcbsSimulator* sim,
                double Tb,
                double Ts,
                double theta_c,
                double dtheta_c,
                double ddtheta_c,
                TcbsState* out);

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
TcbsTrajectory* tcbs_trajectory_create(const TcbsParams* params, int refinement);

/* 销毁；传入 NULL 是安全的。 */
void tcbs_trajectory_destroy(TcbsTrajectory* t);

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
 * 失败返回 -1 并通过 tcbs_last_error() 给出原因。
 */
double tcbs_trajectory_loss(const TcbsTrajectory* t,
                             double theta_c0,
                             double dtheta_c,
                             double ddtheta_c,
                             double dt,
                             int refinement,
                             size_t num_steps,
                             const double* tau,
                             double tau_b_fixed,
                             double tau_s_fixed,
                             const TcbsState* x0,
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
 * 失败返回 -1 并通过 tcbs_last_error() 给出原因。
 */
double tcbs_trajectory_gradient(TcbsTrajectory* t,
                                 double theta_c0,
                                 double dtheta_c,
                                 double ddtheta_c,
                                 double dt,
                                 int refinement,
                                 size_t num_steps,
                                 const double* tau,
                                 const TcbsState* x0,
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
                                 TcbsState* out_final_state);

/* ==================================================================== */
/* 系统参数辨识模式：对 14 个动力学参数求损失的解析梯度                  */
/* ==================================================================== */

/*
 * 与上面的力矩梯度模式完全独立（不同的句柄、不同的函数），互不影响：
 *   * 力矩序列 tau 是**已知输入**；
 *   * 决策变量是 14 个动力学参数，通过前向灵敏度求导，无需反向扫描。
 *
 * 损失（全局量的四项时间均值加权和，k = 0..K-1）：
 *   L = (1/2K) sum_k [ w_psi_b (psi_b - psi_b*)^2 + w_psi_s (psi_s - psi_s*)^2
 *                    + w_dpsi_b (dpsi_b - dpsi_b*)^2 + w_dpsi_s (dpsi_s - dpsi_s*)^2 ]
 *   psi_b  = theta_c + theta_b          psi_s  = theta_c + theta_b + theta_s
 *   dpsi_b = dtheta_c + dtheta_b        dpsi_s = dtheta_c + dtheta_b + dtheta_s
 *
 * 参数顺序（固定，共 14 个）：
 *   0 mb  1 Ib  2 Pbx  3 Pby  4 ms  5 Is  6 Psx  7 Psy
 *   8 Dx  9 Dy 10 fbc 11 fbv 12 fsc 13 fsv
 *
 * 不参与辨识：gx / gy（重力矢量是随采集数据一起给出的已知输入，
 * 仍保存在 TcbsParams 里供正演使用）、lambda_（固定常数）。
 */

/* 参数个数与名字（用于校验顺序；名字为静态字符串，无需释放）。 */
int tcbs_param_gradient_count(void);
const char* tcbs_param_gradient_name(int index);

/* 创建 / 销毁参数梯度求解器（缓冲可跨调用复用）。 */
TcbsTrajectory* tcbs_param_gradient_create(const TcbsParams* params, int refinement);
void tcbs_param_gradient_destroy(TcbsTrajectory* t);

/*
 * 就地更新求导点参数（不重新分配缓冲，refinement 不变）。
 *
 * 优化循环里**必须**在每次更新参数后调用它，否则 loss/梯度会一直停留在
 * 创建时的参数点上——这是"梯度与参数错配"最隐蔽的一种形态。
 * 失败返回 0 并通过 tcbs_last_error() 给出原因。
 */
int tcbs_param_gradient_set_params(TcbsTrajectory* t, const TcbsParams* params);

/* 读回当前求导点参数（可为 NULL 表示不需要）。 */
void tcbs_param_gradient_get_params(const TcbsTrajectory* t, TcbsParams* out);

/*
 * 仅前向：计算损失值，并可导出四个全局量序列（out_* 可为 NULL，长度 K）。
 * 失败返回 -1 并通过 tcbs_last_error() 给出原因。
 */
double tcbs_param_gradient_loss(const TcbsTrajectory* t,
                                 double theta_c0,
                                 double dtheta_c,
                                 double ddtheta_c,
                                 double dt,
                                 int refinement,
                                 size_t num_steps,
                                 const double* tau,
                                 const TcbsState* x0,
                                 double w_psi_b,
                                 double w_psi_s,
                                 double w_dpsi_b,
                                 double w_dpsi_s,
                                 const double* target_psi_b,
                                 const double* target_psi_s,
                                 const double* target_dpsi_b,
                                 const double* target_dpsi_s,
                                 double* out_psi_b,
                                 double* out_psi_s,
                                 double* out_dpsi_b,
                                 double* out_dpsi_s);

/*
 * 正向 + 前向参数灵敏度：返回损失，并写出 dL/dp（长度 14）。
 * grad_p 不可为 NULL。out_final_state 可为 NULL。
 */
double tcbs_param_gradient_run(TcbsTrajectory* t,
                                double theta_c0,
                                double dtheta_c,
                                double ddtheta_c,
                                double dt,
                                int refinement,
                                size_t num_steps,
                                const double* tau,
                                const TcbsState* x0,
                                double w_psi_b,
                                double w_psi_s,
                                double w_dpsi_b,
                                double w_dpsi_s,
                                const double* target_psi_b,
                                const double* target_psi_s,
                                const double* target_dpsi_b,
                                const double* target_dpsi_s,
                                double* grad_p,
                                TcbsState* out_final_state);


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CAPI_DM_H */

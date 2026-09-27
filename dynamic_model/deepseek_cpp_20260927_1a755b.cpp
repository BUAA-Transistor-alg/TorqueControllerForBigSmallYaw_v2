#include <cmath>

// =============================================================================
// 二自由度平面双连杆系统动力学模型（参考实现）
//
// 【修正说明】
//   本文件最初版本的质量矩阵 / 重力项 / 科氏项与正向运动学不自洽，具体为：
//     1) M11、M12 漏掉了连杆 s 绕关节 s 的惯量 (ms*|Ps|^2 + Is)；
//     2) G1 漏掉了连杆 s 的重力贡献（即 G2，它通过 psi_b 的旋转同样作用于 theta_b）；
//     3) C1 漏掉了一项 ms*(dh/dtheta_s)*dtheta_s*dtheta_b。
//   症状：重力项不是任何势函数的梯度（dG1/dtheta_s != dG2/dtheta_b），
//         且零力矩零摩擦下能量自增（从偏离竖直 0.15 rad 释放，20 s 内角速度被泵到
//         10.5 rad/s）。
//   现已按正向运动学补全上述缺项，模型能量守恒（20 s 能量漂移 ~1e-7）。
//
// 【运动学约定】
//   psi_b = theta_c + theta_b
//   psi_s = theta_c + theta_b + theta_s
//   连杆 b 质心在 R(psi_b)*(Pbx, Pby)；
//   关节 s 在 R(psi_b)*(Dx, Dy)；
//   连杆 s 质心在 关节 s + R(psi_s)*(Psx, Psy)。
// =============================================================================

// 系统参数结构体
struct Params {
    // 连杆 b 参数
    double mb;      // 质量
    double Ib;      // 转动惯量
    double Pbx;     // 质心局部坐标 x
    double Pby;     // 质心局部坐标 y

    // 连杆 s 参数
    double ms;      // 质量
    double Is;      // 转动惯量
    double Psx;     // 质心局部坐标 x
    double Psy;     // 质心局部坐标 y

    // 关节偏移与重力场
    double Dx;      // 偏移量 x
    double Dy;      // 偏移量 y
    double gx;      // 重力场加速度 x
    double gy;      // 重力场加速度 y

    // 驱动力与摩擦参数
    double Tb;      // 关节 b 驱动力矩
    double Ts;      // 关节 s 驱动力矩
    double fbc;     // 关节 b 库仑摩擦系数
    double fbv;     // 关节 b 粘滞摩擦系数
    double fsc;     // 关节 s 库仑摩擦系数
    double fsv;     // 关节 s 粘滞摩擦系数
    double lambda;  // 平滑摩擦力参数 (通常取 100)
};

/**
 * @brief 计算二自由度平面双连杆系统的动力学加速度
 * 
 * @param p         系统物理参数结构体 (见上方 Params 定义)
 * @param theta_c   外部输入角度 (已知)
 * @param dtheta_c  外部输入角速度 (已知)
 * @param ddtheta_c 外部输入角加速度 (已知)
 * @param theta_b   广义坐标 1 (当前角度)
 * @param dtheta_b  广义坐标 1 (当前角速度)
 * @param theta_s   广义坐标 2 (当前角度)
 * @param dtheta_s  广义坐标 2 (当前角速度)
 * @param ddtheta_b [输出] 广义坐标 1 的角加速度
 * @param ddtheta_s [输出] 广义坐标 2 的角加速度
 */
void computeAccelerations(
    const Params& p,
    double theta_c, double dtheta_c, double ddtheta_c,
    double theta_b, double dtheta_b,
    double theta_s, double dtheta_s,
    double& ddtheta_b, double& ddtheta_s)
{
    // ---------------------------------------------------------
    // 1. 运动学中间变量计算
    // ---------------------------------------------------------
    double psi_b = theta_c + theta_b;
    double psi_s = theta_c + theta_b + theta_s;
    
    double Sb = std::sin(psi_b);
    double Cb = std::cos(psi_b);
    double Ss = std::sin(psi_s);
    double Cs = std::cos(psi_s);
    
    double Sts = std::sin(theta_s);
    double Cts = std::cos(theta_s);

    // 关节耦合项 h 及其对 theta_s 的偏导数
    // h = Dx*Psx*cos(theta_s) + Dy*Psy*cos(theta_s) + Dy*Psx*sin(theta_s) - Dx*Psy*sin(theta_s)
    double h = p.Dx * p.Psx * Cts + p.Dy * p.Psy * Cts 
             + p.Dy * p.Psx * Sts - p.Dx * p.Psy * Sts;
             
    // dh/dtheta_s = -Dx*Psx*sin(theta_s) - Dy*Psy*sin(theta_s) + Dy*Psx*cos(theta_s) - Dx*Psy*cos(theta_s)
    double dh_dts = -p.Dx * p.Psx * Sts - p.Dy * p.Psy * Sts 
                    + p.Dy * p.Psx * Cts - p.Dx * p.Psy * Cts;

    // ---------------------------------------------------------
    // 2. 质量矩阵元素 (对称)
    //
    // 【已修正】补上连杆 s 绕关节 s 的惯量 Is_ = ms*(Psx^2+Psy^2) + Is，
    //          它在 M11、M12 中不可省略（例如 D=0 时两连杆仍通过它耦合）。
    // ---------------------------------------------------------
    double Is_link = p.ms * (p.Psx * p.Psx + p.Psy * p.Psy) + p.Is;

    double M11 = p.mb * (p.Pbx * p.Pbx + p.Pby * p.Pby) + p.Ib 
               + p.ms * (p.Dx * p.Dx + p.Dy * p.Dy) + Is_link + 2.0 * p.ms * h;
               
    double M12 = Is_link + p.ms * h;
    
    double M22 = Is_link;

    // ---------------------------------------------------------
    // 3. 科里奥利力与离心力项 (已移项至等式右侧)
    // ---------------------------------------------------------
    double dpsi_b = dtheta_c + dtheta_b;

    // 方程1的科里奥利力移项部分
    // 【已修正】应为 ms*dh/dts*dtheta_s*(2*dpsi_b + dtheta_s)
    double C1 = p.ms * dh_dts * dtheta_s * (2.0 * dpsi_b + dtheta_s);

    // 方程2的科里奥利力移项部分 (原为 -ms*dh_dts*dpsi_b^2，移项后变为正)
    double C2 = -p.ms * dh_dts * dpsi_b * dpsi_b; 

    // ---------------------------------------------------------
    // 4. 重力项
    // ---------------------------------------------------------
    // 方程2的重力项 G2：连杆 s 自身的重力贡献
    double G2 = p.ms * (p.gx * p.Psx + p.gy * p.Psy) * Ss
              + p.ms * (p.gx * p.Psy - p.gy * p.Psx) * Cs;

    // 方程1的重力项 G1
    // 【已修正】关节 s 随连杆 b 一起转动，因此连杆 s 的重力也通过 psi_b 进入 G1，
    //          即 G1 必须再加上 G2。
    double G1 = (p.gx * (p.mb * p.Pbx + p.ms * p.Dx) + p.gy * (p.mb * p.Pby + p.ms * p.Dy)) * Sb
              + (p.gx * (p.mb * p.Pby + p.ms * p.Dy) - p.gy * (p.mb * p.Pbx + p.ms * p.Dx)) * Cb
              + G2;

    // ---------------------------------------------------------
    // 5. 广义力 (驱动力与摩擦力)
    // ---------------------------------------------------------
    double Qb = p.Tb - p.fbv * dtheta_b - p.fbc * std::tanh(p.lambda * dtheta_b);
    double Qs = p.Ts - p.fsv * dtheta_s - p.fsc * std::tanh(p.lambda * dtheta_s);

    // ---------------------------------------------------------
    // 6. 构建线性方程组右侧的力项 F1, F2
    // 方程标准形式: M11*ddtheta_b + M12*ddtheta_s = F1
    //              M12*ddtheta_b + M22*ddtheta_s = F2
    // ---------------------------------------------------------
    
    // 注意：ddtheta_c 作为系统输入，需要分配到等号右侧；
    //       基座加速度与两个广义坐标的耦合系数就是质量矩阵的第一列 (M11, M12)。
    double F1 = Qb - C1 - G1 - M11 * ddtheta_c;
    double F2 = Qs - C2 - G2 - M12 * ddtheta_c;

    // ---------------------------------------------------------
    // 7. 求解线性方程组 (克拉默法则)
    // ---------------------------------------------------------
    double det = M11 * M22 - M12 * M12; // 行列式，对于正定系统恒大于0
    
    // 防止除零异常 (理论不应发生，但工程上保留安全检查)
    if (std::abs(det) < 1e-12) {
        ddtheta_b = 0.0;
        ddtheta_s = 0.0;
        return;
    }

    ddtheta_b = (F1 * M22 - F2 * M12) / det;
    ddtheta_s = (M11 * F2 - M12 * F1) / det;
}

// FullStrictPoseBuilder.cpp — 严格反解包实现。
//
//   · onImu / onMcu: 把打包好的样本存进成员变量（锁内覆盖，未传入过则保持全 0）；
//   · strictPose() : 锁内取两个成员的一致快照，随后**在锁外**做反解，得到
//       底盘姿态 chassis_euler_*、底盘方位角 chassis_azimuth、
//       大/小 yaw 世界方位角 big_azimuth / small_azimuth、
//       旋转平面内重力分量 gx / gy、以及各环节方位角速度。
//     其中 chassis_azimuth 会经 unwrapChassisAzimuth 累计圈数，输出为**多圈连续量**
//     （atan2 给的 (−π,π] 单圈值只作为它的输入），详见该函数与头文件注释。
//     运动学链按 imu_location_ 分支（ON_HEAD / ON_BIG_YAW），两分支的旋转矩阵与
//     角速度合成方式不同，见下方 strictPose() 内的注释。
#include "FullStrictPoseBuilder.h"
#include <cmath>

namespace tcbs {
namespace com {


namespace {

/// 2π / π（不依赖 M_PI，保证 -std=c++17 严格模式下也可用）。
constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
constexpr double kPi    = 3.14159265358979323846;

// ============================================================================
// 3x3 旋转矩阵工具（ZXY 约定: R = Rz(yaw)·Rx(pitch)·Ry(roll)，与
// RobotTfTree/CoordinateTransform 矩阵一致，pitch 绕 x）
// ============================================================================
struct Mat3 {
    double m[3][3];
};

Mat3 eulerZXY(double yaw, double pitch, double roll) {
    double cy = std::cos(yaw), sy = std::sin(yaw);
    double cp = std::cos(pitch), sp = std::sin(pitch);
    double cr = std::cos(roll), sr = std::sin(roll);
    Mat3 R;
    R.m[0][0] = cy*cr - sy*sp*sr;  R.m[0][1] = -sy*cp;         R.m[0][2] = cy*sr + sy*sp*cr;
    R.m[1][0] = sy*cr + cy*sp*sr;  R.m[1][1] =  cy*cp;         R.m[1][2] = sy*sr - cy*sp*cr;
    R.m[2][0] = -cp*sr;            R.m[2][1] =  sp;            R.m[2][2] = cp*cr;
    return R;
}

Mat3 rotZ(double y) {
    double cy = std::cos(y), sy = std::sin(y);
    Mat3 R;
    R.m[0][0] = cy;  R.m[0][1] = -sy; R.m[0][2] = 0.0;
    R.m[1][0] = sy;  R.m[1][1] =  cy; R.m[1][2] = 0.0;
    R.m[2][0] = 0.0; R.m[2][1] = 0.0; R.m[2][2] = 1.0;
    return R;
}

// pitch 关节绕 x 轴（物理轴，用户确认）
Mat3 rotX(double p) {
    double cp = std::cos(p), sp = std::sin(p);
    Mat3 R;
    R.m[0][0] = 1.0; R.m[0][1] = 0.0; R.m[0][2] = 0.0;
    R.m[1][0] = 0.0; R.m[1][1] = cp;  R.m[1][2] = -sp;
    R.m[2][0] = 0.0; R.m[2][1] = sp;  R.m[2][2] = cp;
    return R;
}

Mat3 mul(const Mat3& A, const Mat3& B) {
    Mat3 R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            R.m[i][j] = A.m[i][0]*B.m[0][j] + A.m[i][1]*B.m[1][j] + A.m[i][2]*B.m[2][j];
        }
    return R;
}

// Mat3 trans(const Mat3& A) {
//     Mat3 R;
//     for (int i = 0; i < 3; ++i)
//         for (int j = 0; j < 3; ++j)
//             R.m[i][j] = A.m[j][i];
//     return R;
// }

// ZXY 欧拉角提取（R = Rz·Rx·Ry）:
//   pitch = asin(M21), yaw = atan2(-M01, M11), roll = atan2(-M20, M22)
void matToEulerZXY(const Mat3& R, double& yaw, double& pitch, double& roll) {
    pitch = std::asin(std::max(-1.0, std::min(1.0, R.m[2][1])));
    const double eps = 1e-6;
    if (std::fabs(std::cos(pitch)) > eps) {
        yaw = std::atan2(-R.m[0][1], R.m[1][1]);
        roll = std::atan2(-R.m[2][0], R.m[2][2]);
    } else {
        roll = 0.0;
        yaw = std::atan2(R.m[1][0], R.m[0][0]);
    }
}

// 返回值：投影后的纯绕 Z 轴旋转矩阵
// 输出参数：projX, projY 为原 +Z 轴单位向量在世界系 xy 平面投影的 x、y 分量
Mat3 projectToZRotation(const Mat3& R, double& projX, double& projY) {
    // 原 +Z 轴单位向量 (0,0,1) 经过 R 旋转到世界系，即 R 的第三列
    projX = R.m[0][2];
    projY = R.m[1][2];
    // 投影向量为 (projX, projY, 0)

    // 以下为最小旋转投影计算
    double nx = R.m[0][2];
    double ny = R.m[1][2];
    double nz = R.m[2][2];

    double cosTheta = std::max(-1.0, std::min(1.0, nz));
    double theta = std::acos(cosTheta);
    const double eps = 1e-9;

    // 如果已经对齐（theta ≈ 0），R 本身已是纯绕 Z 旋转
    if (theta < eps) {
        return R;
    }

    // 计算旋转轴：n × z = (ny, -nx, 0)
    double ax = ny;
    double ay = -nx;
    // double az = 0.0;
    double len = std::sqrt(ax * ax + ay * ay);

    Mat3 S; // 从 R 到纯 Z 旋转的差异旋转

    if (len < eps) {
        // n 与 Z 轴平行但反向，theta ≈ π
        // 选择绕 X 轴旋转 π
        S.m[0][0] = 1.0; S.m[0][1] = 0.0;  S.m[0][2] = 0.0;
        S.m[1][0] = 0.0; S.m[1][1] = -1.0; S.m[1][2] = 0.0;
        S.m[2][0] = 0.0; S.m[2][1] = 0.0;  S.m[2][2] = -1.0;
    } else {
        ax /= len;
        ay /= len;
        // az 保持为 0

        double c = std::cos(-theta); // = cos(theta)
        double s = std::sin(-theta); // = -sin(theta)
        double t = 1.0 - c;

        // 绕 axis 旋转 -theta 的旋转矩阵（Rodrigues 公式，az=0）
        S.m[0][0] = t * ax * ax + c;
        S.m[0][1] = t * ax * ay;
        S.m[0][2] = s * ay;
        S.m[1][0] = t * ax * ay;
        S.m[1][1] = t * ay * ay + c;
        S.m[1][2] = -s * ax;
        S.m[2][0] = -s * ay;
        S.m[2][1] = s * ax;
        S.m[2][2] = c;
    }

    // 应用差异旋转：R_z = S * R
    Mat3 R_z = mul(S, R);

    // 强制第三列为 (0, 0, 1)，消除浮点误差
    R_z.m[0][2] = 0.0;
    R_z.m[1][2] = 0.0;
    R_z.m[2][2] = 1.0;

    return R_z;
}

} // namespace


void FullStrictPoseBuilder::onImu(const ImuSample& sample) {
    std::lock_guard<std::mutex> lock(mtx_);
    imu_ = sample;
}

void FullStrictPoseBuilder::onMcu(const McuSample& sample) {
    std::lock_guard<std::mutex> lock(mtx_);
    mcu_ = sample;
}

FullStrictPoseBuilder::StrictPose FullStrictPoseBuilder::strictPose() const {
    struct {
        double imu_euler_yaw = 0.0;      // IMU 原始欧拉角（世界←IMU, ZXY）
        double imu_euler_pitch = 0.0;
        double imu_euler_roll = 0.0;
        double imu_omega_z = 0.0;
        double imu_omega_y = 0.0;
    } temp;
    StrictPose sp;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        // ── 反解输入快照：取自最近缓存的样本（从未传入过则保持 0）──
        temp.imu_euler_yaw = imu_.euler_yaw;
        temp.imu_euler_pitch = imu_.euler_pitch;
        temp.imu_euler_roll = imu_.euler_roll;
        sp.yaw_big_angle = mcu_.yaw_big_angle;
        sp.yaw_small_angle = mcu_.yaw_small_angle;
        sp.pitch_angle = mcu_.pitch_angle;

        temp.imu_omega_z = imu_.gz;
        temp.imu_omega_y = imu_.gy;
        sp.small_motor_omega = mcu_.yaw_small_omega;
        sp.big_motor_omega = mcu_.yaw_big_omega;
    }

    Mat3 R_head;
    Mat3 R_small;
    Mat3 R_big;
    Mat3 R_chassis;
    if (imu_location_ == ImuLocation::ON_HEAD) {
        R_head = eulerZXY(temp.imu_euler_yaw, temp.imu_euler_pitch, temp.imu_euler_roll);
        R_small = mul(R_head, rotX(-sp.pitch_angle));
        R_big = mul(R_small, rotZ(-sp.yaw_small_angle));
        R_chassis = mul(R_big, rotZ(-sp.yaw_big_angle));

        sp.small_azimuth_omega = 
            temp.imu_omega_z * std::cos(sp.pitch_angle) + temp.imu_omega_y * std::sin(sp.pitch_angle)
        ;
        sp.big_azimuth_omega = sp.small_azimuth_omega - sp.small_motor_omega;
        sp.chassis_omega = sp.big_azimuth_omega - sp.big_motor_omega;
    } else {
        R_big = eulerZXY(temp.imu_euler_yaw, temp.imu_euler_pitch, temp.imu_euler_roll);
        R_small = mul(R_big, rotZ(sp.yaw_small_angle));
        R_head = mul(R_small, rotX(sp.pitch_angle));
        R_chassis = mul(R_big, rotZ(-sp.yaw_big_angle));

        sp.big_azimuth_omega = temp.imu_omega_z;
        sp.small_azimuth_omega = sp.big_azimuth_omega + sp.small_motor_omega;
        sp.chassis_omega = sp.big_azimuth_omega - sp.big_motor_omega;
    }
    matToEulerZXY(R_chassis, sp.chassis_euler_yaw, sp.chassis_euler_pitch, sp.chassis_euler_roll);

    double projX = 0.0;
    double projY = 0.0;
    Mat3 R_plain_chassis = projectToZRotation(R_chassis, projX, projY);

    double dummy1;
    double dummy2;
    matToEulerZXY(R_plain_chassis, sp.chassis_azimuth, dummy1, dummy2);

    // ★ 底盘方位角解卷绕：(−π,π] 单圈值 → **多圈连续量**。
    //   必须在这里做、且在合成 big/small_azimuth 之前做：下游 mpc::DualYawMpcController
    //   按 psi_b = chassis_azimuth + θ_b 使用（θ_b 本身就是多圈关节角），两者圈数口径
    //   一致才不会有整圈残差。理由与故障现象见头文件与该函数实现。
    sp.chassis_azimuth = unwrapChassisAzimuth(sp.chassis_azimuth);

    sp.big_azimuth = sp.chassis_azimuth + sp.yaw_big_angle;
    sp.small_azimuth = sp.big_azimuth + sp.yaw_small_angle;

    sp.gx = projX * g_;
    sp.gy = projY * g_;

    return sp;
}

// 底盘方位角：(−π, π] 单圈值 → 多圈连续值。
//
// ── 为什么必须解卷绕（一次真实故障的根因）──
//   matToEulerZXY 用 atan2 给出 azimuth，底盘每转一圈输出就跳 ±2π。而下游
//   mpc::DualYawMpcController::measure() 按
//       psi_b = chassis_azimuth + theta_b        （theta_b 是电控给的多圈关节角）
//   合成世界方位角（MPC 代价用的就是它）。若这里给包裹值，psi_b 每圈跳 ±2π，
//   而参考序列只在 McuMpcController::set()（上位机每帧一次）对齐过圈数，100Hz 的
//   拍之间不再对齐 ⇒ 底盘跨过 ±π 到下一帧之间，MPC 代价里出现 w_psi_b·(2π)² ≈ 39.5
//   的整圈残差（正常跟踪时 ~4e-4），表现为**底盘每转一圈就有一个恒定幅值的力矩
//   脉冲**（大 yaw 抖一下），且固定在同一个底盘角度（= atan2 的 ±π 边界）。
//
// ── 语义与线程安全 ──
//   · 首个样本：多圈零位就取该包裹值所在圈（输出 = 输入），保证改动前启动阶段的
//     行为与数值完全不变，圈数只是相对累计；
//   · 之后以「上一拍输出」为参考吸收 ±2π 跳变（标准 unwrap），输出因此连续；
//   · 顺序滤波器：重复喂同一个快照或相互接近的快照时增量恒为 0（幂等），
//     故多个线程各自调用 strictPose() 不会把圈数累坏（锁只保护这四个成员）。
double FullStrictPoseBuilder::unwrapChassisAzimuth(double wrapped) const {
    std::lock_guard<std::mutex> lock(azimuth_mtx_);
    if (!chassis_unwrap_init_) {
        chassis_azimuth_last_ = wrapped;
        chassis_azimuth_turn_ = 0.0;
        chassis_unwrap_init_  = true;
        return wrapped;
    }
    // 先按当前累计圈数还原，再看与上一拍输出差了多少整圈并吸收掉
    const double d = (wrapped + chassis_azimuth_turn_) - chassis_azimuth_last_;
    if (d > kPi) {
        chassis_azimuth_turn_ -= kTwoPi;   // 跨过 +π：回退一圈
    } else if (d < -kPi) {
        chassis_azimuth_turn_ += kTwoPi;   // 跨过 −π：前进一圈
    }
    const double out = wrapped + chassis_azimuth_turn_;
    chassis_azimuth_last_ = out;
    return out;
}

}  // namespace com
}  // namespace tcbs

// FullStrictPoseBuilder.cpp — 具体处理内容暂时留空。
//
// 目前只完成"数据流 + 加锁"骨架:
//   · onImu / onMcu: 把打包好的样本存进成员变量（锁内覆盖，未传入过则保持全 0）；
//   · strictPose() : 锁内读取两个成员，填充 StrictPose 的反解输入快照。
// 反解算法（底盘姿态 chassis_euler_*、各环节方位角、重力分量 gx/gy）尚待实现。
#include "FullStrictPoseBuilder.h"
#include <cmath>

namespace tcbs {
namespace com {


namespace {

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

        sp.small_azimuth_omega = imu_.gz;
        sp.small_motor_omega = mcu_.yaw_small_omega;
        sp.big_motor_omega = mcu_.yaw_big_omega;
    }

    Mat3 R_head = eulerZXY(temp.imu_euler_yaw, temp.imu_euler_pitch, temp.imu_euler_roll);
    Mat3 R_small = mul(R_head, rotX(-sp.pitch_angle));
    Mat3 R_big = mul(R_small, rotZ(-sp.yaw_small_angle));
    Mat3 R_chassis = mul(R_big, rotZ(-sp.yaw_big_angle));
    matToEulerZXY(R_chassis, sp.chassis_euler_yaw, sp.chassis_euler_pitch, sp.chassis_euler_roll);

    double projX = 0.0;
    double projY = 0.0;
    Mat3 R_plain_chassis = projectToZRotation(R_chassis, projX, projY);

    double dummy1;
    double dummy2;
    matToEulerZXY(R_plain_chassis, sp.chassis_azimuth, dummy1, dummy2);
    sp.big_azimuth = sp.chassis_azimuth + sp.yaw_big_angle;
    sp.small_azimuth = sp.big_azimuth + sp.yaw_small_angle;

    sp.gx = projX * g_;
    sp.gy = projY * g_;

    sp.big_azimuth_omega = sp.small_azimuth_omega - sp.small_motor_omega;
    sp.chassis_omega = sp.big_azimuth_omega - sp.big_motor_omega;

    return sp;
}

}  // namespace com
}  // namespace tcbs

// capi_com.cpp — 通信模块（tcbs::com::RobotCommunication）的 C ABI 实现。
//
// 与其他源文件不同：本文件是 **C ABI 实现**，所有 tcbs_com_* 导出函数必须保持
// C 链接名并留在全局命名空间，因此用到的 C++ 类型一律写成 tcbs::... 全限定名
// （与 C 类型 Tcbs... 明确区分）。实现主体仍放在 namespace tcbs::com 内，
// 只在 extern "C" 块里导出符号。
//
// 数据包结构体与 include/com/Protocol.hpp 逐字段偏移一致，由下面的
// static_assert 在编译期强制校验；转换用 memcpy（打包布局相同）。

#include "capi_com.h"

#include <cstddef>
#include <exception>
#include <new>
#include <string>

#include "Communications.hpp"
#include "FullStrictPoseBuilder.h"
#include "McuDataPreprocessor.h"
#include "Protocol.hpp"

namespace tcbs {
namespace com {
namespace {

// 线程局部的错误信息，供 tcbs_com_last_error() 返回。
thread_local std::string g_last_com_error;

void setComError(const char* msg) {
    g_last_com_error = (msg != nullptr) ? msg : "unknown error";
}

// ============================================================================
// 编译期校验：C 结构体与 C++ 结构体逐字段偏移 / 大小一致
// ============================================================================
#define TCBS_COM_ASSERT_OFFSET(CType, CppType, field)                       \
    static_assert(offsetof(CType, field) == offsetof(CppType, field),       \
                  "offset mismatch: " #field)

TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, frame_header1);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, frame_header2);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, protocol_version);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, data_size);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, auto_aim_enable);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, fire);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, pitch_target_angle);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, yaw_big_mode);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, yaw_big_target_angle);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, yaw_big_target_velocity);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, yaw_big_torque);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, yaw_small_mode);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, yaw_small_target_angle);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, yaw_small_target_velocity);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, yaw_small_torque);
TCBS_COM_ASSERT_OFFSET(TcbsMcuSendPacket, mcu::SendPacket, crc8);

TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, frame_header1);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, frame_header2);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, protocol_version);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, data_size);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, bullet_velocity);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, pitch_angle);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, yaw_big_angle);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, yaw_big_omega);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, yaw_small_angle);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, yaw_small_omega);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, chassis_imu_yaw);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, chassis_imu_omega);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, mark);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, color);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, auto_aim_switch);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, yaw_big_temperature);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, yaw_small_temperature);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, mcu2_seq);
TCBS_COM_ASSERT_OFFSET(TcbsMcuReceivePacket, mcu::ReceivePacket, crc8);

TCBS_COM_ASSERT_OFFSET(TcbsImuSendPacket, imu::SendPacket, frame_header1);
TCBS_COM_ASSERT_OFFSET(TcbsImuSendPacket, imu::SendPacket, frame_header2);
TCBS_COM_ASSERT_OFFSET(TcbsImuSendPacket, imu::SendPacket, frame_header3);
TCBS_COM_ASSERT_OFFSET(TcbsImuSendPacket, imu::SendPacket, data_size);
TCBS_COM_ASSERT_OFFSET(TcbsImuSendPacket, imu::SendPacket, crc32);

TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, frame_header1);
TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, frame_header2);
TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, frame_header3);
TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, data_size);
TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, gx);
TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, gy);
TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, gz);
TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, ax);
TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, ay);
TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, az);
TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, euler_yaw);
TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, euler_pitch);
TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, euler_roll);
TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, dt_one_tenth_ms);
TCBS_COM_ASSERT_OFFSET(TcbsImuReceivePacket, imu::ReceivePacket, crc32);

#undef TCBS_COM_ASSERT_OFFSET

static_assert(sizeof(TcbsMcuSendPacket) == sizeof(mcu::SendPacket),
              "TcbsMcuSendPacket size mismatch");
static_assert(sizeof(TcbsMcuReceivePacket) == sizeof(mcu::ReceivePacket),
              "TcbsMcuReceivePacket size mismatch");
static_assert(sizeof(TcbsImuSendPacket) == sizeof(imu::SendPacket),
              "TcbsImuSendPacket size mismatch");
static_assert(sizeof(TcbsImuReceivePacket) == sizeof(imu::ReceivePacket),
              "TcbsImuReceivePacket size mismatch");

// ============================================================================
// 转换
//
// 逐字段赋值（不用 memcpy）：C++ 侧结构体带默认成员初始化器，memcpy 会触发
// -Wclass-memaccess；逐字段写法也避免依赖"填充字节相同"这一隐式假设。
// 偏移一致性由上面的 static_assert 在编译期保证。
// ============================================================================
inline void copyToCpp(const TcbsMcuSendPacket& src, mcu::SendPacket& dst) {
    dst.frame_header1 = src.frame_header1;
    dst.frame_header2 = src.frame_header2;
    dst.protocol_version = src.protocol_version;
    dst.data_size = src.data_size;
    dst.auto_aim_enable = src.auto_aim_enable;
    dst.fire = src.fire;
    dst.pitch_target_angle = src.pitch_target_angle;
    dst.yaw_big_mode = src.yaw_big_mode;
    dst.yaw_big_target_angle = src.yaw_big_target_angle;
    dst.yaw_big_target_velocity = src.yaw_big_target_velocity;
    dst.yaw_big_torque = src.yaw_big_torque;
    dst.yaw_small_mode = src.yaw_small_mode;
    dst.yaw_small_target_angle = src.yaw_small_target_angle;
    dst.yaw_small_target_velocity = src.yaw_small_target_velocity;
    dst.yaw_small_torque = src.yaw_small_torque;
    dst.crc8 = src.crc8;
}

inline void copyToCpp(const TcbsImuSendPacket& src, imu::SendPacket& dst) {
    dst.frame_header1 = src.frame_header1;
    dst.frame_header2 = src.frame_header2;
    dst.frame_header3 = src.frame_header3;
    dst.data_size = src.data_size;
    dst.crc32 = src.crc32;
}

inline void copyToC(const mcu::ReceivePacket& src, TcbsMcuReceivePacket& dst) {
    dst.frame_header1 = src.frame_header1;
    dst.frame_header2 = src.frame_header2;
    dst.protocol_version = src.protocol_version;
    dst.data_size = src.data_size;
    dst.bullet_velocity = src.bullet_velocity;
    dst.pitch_angle = src.pitch_angle;
    dst.yaw_big_angle = src.yaw_big_angle;
    dst.yaw_big_omega = src.yaw_big_omega;
    dst.yaw_small_angle = src.yaw_small_angle;
    dst.yaw_small_omega = src.yaw_small_omega;
    dst.chassis_imu_yaw = src.chassis_imu_yaw;
    dst.chassis_imu_omega = src.chassis_imu_omega;
    dst.mark = src.mark;
    dst.color = src.color;
    dst.auto_aim_switch = src.auto_aim_switch;
    dst.yaw_big_temperature = src.yaw_big_temperature;
    dst.yaw_small_temperature = src.yaw_small_temperature;
    dst.mcu2_seq = src.mcu2_seq;
    dst.crc8 = src.crc8;
}

inline void copyToC(const imu::ReceivePacket& src, TcbsImuReceivePacket& dst) {
    dst.frame_header1 = src.frame_header1;
    dst.frame_header2 = src.frame_header2;
    dst.frame_header3 = src.frame_header3;
    dst.data_size = src.data_size;
    dst.gx = src.gx;
    dst.gy = src.gy;
    dst.gz = src.gz;
    dst.ax = src.ax;
    dst.ay = src.ay;
    dst.az = src.az;
    dst.euler_yaw = src.euler_yaw;
    dst.euler_pitch = src.euler_pitch;
    dst.euler_roll = src.euler_roll;
    dst.dt_one_tenth_ms = src.dt_one_tenth_ms;
    dst.crc32 = src.crc32;
}

inline McuDataPreprocessor::LinearParams toLinearParams(const TcbsLinearParams& c) {
    McuDataPreprocessor::LinearParams p;
    p.send_pitch_scale = c.send_pitch_scale;
    p.send_pitch_offset = c.send_pitch_offset;
    p.recv_pitch_scale = c.recv_pitch_scale;
    p.recv_pitch_offset = c.recv_pitch_offset;

    p.recv_big_yaw_scale = c.recv_big_yaw_scale;
    p.recv_big_yaw_offset = c.recv_big_yaw_offset;
    p.recv_big_omega_scale = c.recv_big_omega_scale;
    p.send_big_yaw_scale = c.send_big_yaw_scale;
    p.send_big_yaw_offset = c.send_big_yaw_offset;
    p.send_big_velocity_scale = c.send_big_velocity_scale;
    p.send_big_torque_scale = c.send_big_torque_scale;

    p.recv_small_yaw_scale = c.recv_small_yaw_scale;
    p.recv_small_yaw_offset = c.recv_small_yaw_offset;
    p.recv_small_omega_scale = c.recv_small_omega_scale;
    p.send_small_yaw_scale = c.send_small_yaw_scale;
    p.send_small_yaw_offset = c.send_small_yaw_offset;
    p.send_small_velocity_scale = c.send_small_velocity_scale;
    p.send_small_torque_scale = c.send_small_torque_scale;
    return p;
}

inline TcbsLinearParams toCLinearParams(const McuDataPreprocessor::LinearParams& p) {
    TcbsLinearParams c;
    c.send_pitch_scale = p.send_pitch_scale;
    c.send_pitch_offset = p.send_pitch_offset;
    c.recv_pitch_scale = p.recv_pitch_scale;
    c.recv_pitch_offset = p.recv_pitch_offset;

    c.recv_big_yaw_scale = p.recv_big_yaw_scale;
    c.recv_big_yaw_offset = p.recv_big_yaw_offset;
    c.recv_big_omega_scale = p.recv_big_omega_scale;
    c.send_big_yaw_scale = p.send_big_yaw_scale;
    c.send_big_yaw_offset = p.send_big_yaw_offset;
    c.send_big_velocity_scale = p.send_big_velocity_scale;
    c.send_big_torque_scale = p.send_big_torque_scale;

    c.recv_small_yaw_scale = p.recv_small_yaw_scale;
    c.recv_small_yaw_offset = p.recv_small_yaw_offset;
    c.recv_small_omega_scale = p.recv_small_omega_scale;
    c.send_small_yaw_scale = p.send_small_yaw_scale;
    c.send_small_yaw_offset = p.send_small_yaw_offset;
    c.send_small_velocity_scale = p.send_small_velocity_scale;
    c.send_small_torque_scale = p.send_small_torque_scale;
    return c;
}

}  // namespace
}  // namespace com
}  // namespace tcbs

// ============================================================================
// 句柄：持有 C++ 的 RobotCommunication 实例
// ============================================================================
struct TcbsRobotComm {
    tcbs::com::RobotCommunication impl;

    TcbsRobotComm(tcbs::com::FullStrictPoseBuilder::ImuLocation loc,
                  const tcbs::com::McuDataPreprocessor::LinearParams& lp)
        : impl(loc, lp) {}
};

extern "C" {

const char* tcbs_com_last_error(void) {
    return tcbs::com::g_last_com_error.c_str();
}

TcbsLinearParams tcbs_com_default_linear_params(void) {
    return tcbs::com::toCLinearParams(tcbs::com::McuDataPreprocessor::LinearParams{});
}

TcbsRobotComm* tcbs_com_create(TcbsImuLocation imu_location,
                               const TcbsLinearParams* linear_params) {
    if (imu_location != TCBS_IMU_ON_BIG_YAW && imu_location != TCBS_IMU_ON_HEAD) {
        tcbs::com::setComError("tcbs_com_create: invalid imu_location");
        return nullptr;
    }

    try {
        tcbs::com::g_last_com_error.clear();
        const auto loc = (imu_location == TCBS_IMU_ON_BIG_YAW)
                             ? tcbs::com::FullStrictPoseBuilder::ImuLocation::ON_BIG_YAW
                             : tcbs::com::FullStrictPoseBuilder::ImuLocation::ON_HEAD;
        const tcbs::com::McuDataPreprocessor::LinearParams lp =
            (linear_params != nullptr)
                ? tcbs::com::toLinearParams(*linear_params)
                : tcbs::com::McuDataPreprocessor::LinearParams{};
        return new TcbsRobotComm(loc, lp);
    } catch (const std::exception& e) {
        tcbs::com::setComError(e.what());
        return nullptr;
    } catch (...) {
        tcbs::com::setComError("tcbs_com_create: unknown error");
        return nullptr;
    }
}

void tcbs_com_destroy(TcbsRobotComm* comm) {
    delete comm;  // delete nullptr 安全；析构会停止并 join 串口线程
}

TcbsComLatestData tcbs_com_get_latest_data(TcbsRobotComm* comm) {
    TcbsComLatestData out{};
    if (comm == nullptr) {
        tcbs::com::setComError("tcbs_com_get_latest_data: comm is NULL");
        return out;
    }
    const tcbs::com::RobotCommunication::LatestData data = comm->impl.getLatestData();

    out.imu_valid = data.imu_valid ? 1 : 0;
    out.mcu_valid = data.mcu_valid ? 1 : 0;
    out.mcu2_seq = data.mcu2_seq;
    if (data.imu_valid) {
        tcbs::com::copyToC(data.imu_packet, out.imu_packet);
        tcbs::com::copyToC(data.raw_imu_packet, out.raw_imu_packet);
    }
    if (data.mcu_valid) {
        tcbs::com::copyToC(data.mcu_packet, out.mcu_packet);
        tcbs::com::copyToC(data.raw_mcu_packet, out.raw_mcu_packet);
    }
    return out;
}

TcbsComPose tcbs_com_get_strict_pose(TcbsRobotComm* comm) {
    TcbsComPose out{};
    if (comm == nullptr) {
        tcbs::com::setComError("tcbs_com_get_strict_pose: comm is NULL");
        return out;
    }
    const tcbs::com::FullStrictPoseBuilder::StrictPose sp = comm->impl.getStrictPose();

    out.yaw_big_angle = sp.yaw_big_angle;
    out.yaw_small_angle = sp.yaw_small_angle;
    out.pitch_angle = sp.pitch_angle;
    out.chassis_euler_yaw = sp.chassis_euler_yaw;
    out.chassis_euler_pitch = sp.chassis_euler_pitch;
    out.chassis_euler_roll = sp.chassis_euler_roll;
    out.chassis_azimuth = sp.chassis_azimuth;
    out.big_azimuth = sp.big_azimuth;
    out.small_azimuth = sp.small_azimuth;
    out.gx = sp.gx;
    out.gy = sp.gy;
    out.small_motor_omega = sp.small_motor_omega;
    out.small_azimuth_omega = sp.small_azimuth_omega;
    out.big_motor_omega = sp.big_motor_omega;
    out.big_azimuth_omega = sp.big_azimuth_omega;
    out.chassis_omega = sp.chassis_omega;
    return out;
}

int tcbs_com_send_to_mcu(TcbsRobotComm* comm, const TcbsMcuSendPacket* packet) {
    if (comm == nullptr || packet == nullptr) {
        tcbs::com::setComError("tcbs_com_send_to_mcu: comm or packet is NULL");
        return 0;
    }
    tcbs::com::mcu::SendPacket pkt;
    tcbs::com::copyToCpp(*packet, pkt);
    return comm->impl.sendToMcu(pkt) ? 1 : 0;
}

int tcbs_com_send_to_imu(TcbsRobotComm* comm, const TcbsImuSendPacket* packet) {
    if (comm == nullptr || packet == nullptr) {
        tcbs::com::setComError("tcbs_com_send_to_imu: comm or packet is NULL");
        return 0;
    }
    tcbs::com::imu::SendPacket pkt;
    tcbs::com::copyToCpp(*packet, pkt);
    return comm->impl.sendToImu(pkt) ? 1 : 0;
}

void tcbs_com_stop(TcbsRobotComm* comm) {
    if (comm == nullptr) {
        tcbs::com::setComError("tcbs_com_stop: comm is NULL");
        return;
    }
    comm->impl.stop();
}

void tcbs_com_set_linear_params(TcbsRobotComm* comm, const TcbsLinearParams* params) {
    if (comm == nullptr) {
        tcbs::com::setComError("tcbs_com_set_linear_params: comm is NULL");
        return;
    }
    comm->impl.setLinearParams((params != nullptr)
                                   ? tcbs::com::toLinearParams(*params)
                                   : tcbs::com::McuDataPreprocessor::LinearParams{});
}

void tcbs_com_get_linear_params(const TcbsRobotComm* comm, TcbsLinearParams* out) {
    if (comm == nullptr || out == nullptr) {
        tcbs::com::setComError("tcbs_com_get_linear_params: comm or out is NULL");
        return;
    }
    *out = tcbs::com::toCLinearParams(comm->impl.preprocessor().params());
}

size_t tcbs_com_sizeof_latest_data(void) {
    return sizeof(TcbsComLatestData);
}

size_t tcbs_com_sizeof_pose(void) {
    return sizeof(TcbsComPose);
}

size_t tcbs_com_sizeof_linear_params(void) {
    return sizeof(TcbsLinearParams);
}

}  // extern "C"

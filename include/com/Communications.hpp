// Communications.hpp — 基于 SerialProtocol 模板的具体通信类型定义（双级 yaw 版本）
//
// McuCommunication : 与电控（MCU）通信，CRC8，前导 0x42 0x52 0x03
// ImuCommunication : 与大 yaw 上的 IMU 通信，CRC32，前导 0xA7 0xB6 0xC5
//
// RobotCommunication 组合: 两个串口 + McuDataPreprocessor（映射）+ FullStrictPoseBuilder（反解包）
//   - MCU 回调里做映射（processReceive）并缓存"处理前 / 处理后"两份包，随后喂反解包构建器
//   - IMU 回调里缓存最新包（IMU 固定在大 yaw 转子，可信、高频），随后喂反解包构建器
//   - 反解包构建器的接法仿照 TorqueController 中 `YawChassisFusion fusion_` 的用法
//
// 说明（v2 移植）: 本文件由 TorqueControllerForBigSmallYaw 的
//   include/tcbs/communication/Communications.hpp 移植而来，命名空间改为 tcbs::com。
//   原版 RobotCommunication 还组合了 YawStateEstimator（喂状态估计 + getEstimate()），
//   该估计器**不在本次移植范围内**，因此这里已将其移除；其余行为（映射、原始/处理后
//   双份缓存、mcu2_seq 透传、线程安全快照）与原版保持一致。
//   FullStrictPoseBuilder 的具体处理内容暂时留空（见其头文件），此处只固定接法。
#ifndef TCBS_COM_COMMUNICATIONS_HPP
#define TCBS_COM_COMMUNICATIONS_HPP

#include "SerialProtocol.hpp"
#include "Protocol.hpp"
#include "CRC.h"
#include "McuDataPreprocessor.h"
#include "FullStrictPoseBuilder.h"
#include <string>
#include <mutex>

namespace tcbs {
namespace com {

// ── 端口筛选函数 ──
inline bool mcuPortSelector(const std::string& product_info) {
    return product_info != "AutoAim_IMU_Com";
}
inline bool imuPortSelector(const std::string& product_info) {
    return product_info == "AutoAim_IMU_Com";
}

// ── 具体通信类型别名 ──
using McuCommunication = SerialProtocol<
    mcu::SendPacket,
    mcu::ReceivePacket,
    CRC8_Check_Sum,
    mcuPortSelector,
    mcu::PREAMBLE_SIZE
>;

using ImuCommunication = SerialProtocol<
    imu::SendPacket,
    imu::ReceivePacket,
    CRC32_Calculate,
    imuPortSelector,
    imu::PREAMBLE_SIZE
>;

// ============================================================================
// RobotCommunication — 组合 MCU 与 IMU 通信、数据预处理与严格反解包
// ============================================================================
class RobotCommunication {
public:
    struct LatestData {
        bool               imu_valid = false;
        imu::ReceivePacket imu_packet{};
        bool               mcu_valid = false;
        mcu::ReceivePacket mcu_packet{};   // 已映射
        uint8_t            mcu2_seq = 0;               // MCU2 新样本序号（值保持时不变）
        // ★ 新增（追加在末尾，不改已有字段顺序）: **处理前**的原始串口包。
        //   mcu_packet 是 McuDataPreprocessor 的输出；raw_mcu_packet 是它的输入
        //   （采集脚本可据此同时保存"处理前 + 处理后"的数据）。
        //   IMU 不经预处理器，imu_packet 本身即原始包；raw_imu_packet 为对称命名的副本。
        mcu::ReceivePacket raw_mcu_packet{};
        imu::ReceivePacket raw_imu_packet{};
    };

    explicit RobotCommunication(
        const McuDataPreprocessor::LinearParams& mcu_linear_params = McuDataPreprocessor::LinearParams{})
        : preprocessor_(mcu_linear_params)
        , mcu_serial_([this](const mcu::ReceivePacket& pkt) { onMcuReceive(pkt); }, false)
        , imu_serial_([this](const imu::ReceivePacket& pkt) { onImuReceive(pkt); }, false)
        , full_strict_pose_builder_()
    {
        mcu_serial_.startWorker();
        imu_serial_.startWorker();
    }

    ~RobotCommunication() {
        mcu_serial_.stopWorker();
        imu_serial_.stopWorker();
    }

    // 获取最新原始数据（MCU 数据已按映射参数预处理）
    LatestData getLatestData() {
        LatestData data;
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            if (has_imu_data_) {
                data.imu_packet = latest_imu_packet_;
                data.raw_imu_packet = latest_imu_packet_;   // IMU 无预处理 → 与 imu_packet 相同
                data.imu_valid = true;
            }
        }
        {
            std::lock_guard<std::mutex> lock(mcu_mutex_);
            if (has_mcu_data_) {
                data.mcu_packet = latest_mcu_packet_;
                data.raw_mcu_packet = latest_mcu_raw_;      // ★ 处理前原始包
                data.mcu2_seq = latest_mcu2_seq_;
                data.mcu_valid = true;
            }
        }
        return data;
    }

    // ── 原始（处理前）串口包的只读访问器 ──
    // 返回**副本**而非 const&: onXxxReceive 由串口后台线程写入，若返回引用会与写入竞争
    // （调用方拿到的引用可能在读取过程中被后台线程改写）。需要"同一次加锁的一致快照"
    // 时请用 getLatestData()。
    mcu::ReceivePacket lastRawMcuPacket() {
        std::lock_guard<std::mutex> lock(mcu_mutex_);
        return latest_mcu_raw_;
    }
    imu::ReceivePacket lastImuPacket() {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        return latest_imu_packet_;
    }

    // 发送 MCU 数据（发送前按映射参数预处理）
    bool sendToMcu(mcu::SendPacket packet) {
        mcu::SendPacket processed = preprocessor_.processSend(packet);
        return mcu_serial_.sendData(processed);
    }

    // 发送 IMU 数据（心跳等，无预处理）
    bool sendToImu(imu::SendPacket packet) {
        return imu_serial_.sendData(packet);
    }

    void stop() {
        mcu_serial_.stopWorker();
        imu_serial_.stopWorker();
    }

    // 严格反解数据包（全量版；线程安全，始终可读，无 valid 标志）
    FullStrictPoseBuilder::StrictPose getStrictPose() const {
        return full_strict_pose_builder_.strictPose();
    }

    const McuDataPreprocessor& preprocessor() const { return preprocessor_; }
    void setLinearParams(const McuDataPreprocessor::LinearParams& p) { preprocessor_.setParams(p); }

private:
    void onImuReceive(const imu::ReceivePacket& packet) {
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            latest_imu_packet_ = packet;
            has_imu_data_ = true;
        }
        // 喂严格反解包构建器（IMU 不经预处理器，packet 本身即原始数据）
        FullStrictPoseBuilder::ImuSample imu_sample;
        imu_sample.euler_yaw = packet.euler_yaw;
        imu_sample.euler_pitch = packet.euler_pitch;
        imu_sample.euler_roll = packet.euler_roll;
        imu_sample.gx = packet.gx;
        imu_sample.gy = packet.gy;
        imu_sample.gz = packet.gz;
        full_strict_pose_builder_.onImu(imu_sample);
    }

    void onMcuReceive(const mcu::ReceivePacket& packet) {
        // ★ 在 processReceive **之前**保留处理前的原始串口包
        //   （latest_mcu_packet_ 是它的输出；两者供采集脚本"处理前 + 处理后"同时保存）
        {
            std::lock_guard<std::mutex> lock(mcu_mutex_);
            latest_mcu_raw_ = packet;
        }
        const mcu::ReceivePacket processed = preprocessor_.processReceive(packet);
        {
            std::lock_guard<std::mutex> lock(mcu_mutex_);
            latest_mcu_packet_ = processed;
            latest_mcu2_seq_ = packet.mcu2_seq;
            has_mcu_data_ = true;
        }
        // 喂严格反解包构建器的同样为**预处理后**的数据（与 latest_mcu_packet_ 语义一致）
        FullStrictPoseBuilder::McuSample mcu_sample;
        mcu_sample.yaw_big_angle = processed.yaw_big_angle;
        mcu_sample.yaw_big_omega = processed.yaw_big_omega;
        mcu_sample.yaw_small_angle = processed.yaw_small_angle;
        mcu_sample.yaw_small_omega = processed.yaw_small_omega;
        mcu_sample.pitch_angle = processed.pitch_angle;
        mcu_sample.chassis_imu_yaw = processed.chassis_imu_yaw;
        mcu_sample.chassis_imu_omega = processed.chassis_imu_omega;
        full_strict_pose_builder_.onMcu(mcu_sample);
    }

    McuDataPreprocessor  preprocessor_;
    McuCommunication     mcu_serial_;
    ImuCommunication     imu_serial_;

    std::mutex         imu_mutex_;
    imu::ReceivePacket latest_imu_packet_{};
    bool               has_imu_data_ = false;

    std::mutex         mcu_mutex_;
    mcu::ReceivePacket latest_mcu_packet_{};   // McuDataPreprocessor 输出（已映射）
    mcu::ReceivePacket latest_mcu_raw_{};      // ★ 处理前原始串口包（processReceive 输入）
    uint8_t            latest_mcu2_seq_ = 0;
    bool               has_mcu_data_ = false;

    FullStrictPoseBuilder full_strict_pose_builder_;   // IMU 高频 + MCU 低频 → 全量严格反解包
};

}  // namespace com
}  // namespace tcbs

#endif // TCBS_COM_COMMUNICATIONS_HPP

// 通信模块（tcbs::com）自检：不依赖任何硬件。
//
// 覆盖：
//   1. 协议结构体布局 / 前导 / data_size / 关键字段偏移（含 CRC 覆盖范围）；
//   2. CRC8 查表值与独立位运算参考实现逐项一致，函数行为一致；
//   3. CRC32 与独立逐步参考实现一致；
//   4. McuDataPreprocessor 的收发线性映射、平衡判据、符号翻转与透传字段；
//   5. 模板类型（McuCommunication / ImuCommunication / RobotCommunication）可实例化。
//      仅当机器上没有 /dev/ttyACM* 时才真正构造，避免占用真实串口。
//
// 直接运行即可，全部通过返回 0，否则返回 1。
// 构建：cmake 里以 tcbs_com_test 目标生成到 build/ 根目录。

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "CRC.h"
#include "Communications.hpp"
#include "McuDataPreprocessor.h"
#include "Protocol.hpp"
#include "SerialProtocol.hpp"

namespace {

int g_fail = 0;

void check(bool ok, const std::string& name, const std::string& detail = "") {
    std::printf("  [%s] %-56s %s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.c_str());
    if (!ok) ++g_fail;
}

/// 可复现伪随机（避免依赖 <random> 的实现差异）。
struct Rng {
    unsigned long long s = 0x9E3779B97F4A7C15ULL;
    uint8_t nextByte() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return static_cast<uint8_t>(s >> 24);
    }
};

// ── 独立参考实现：按位生成反射多项式 0x8C 的 CRC8 表 ──
void buildCrc8Table(uint8_t out[256]) {
    for (int i = 0; i < 256; ++i) {
        uint8_t crc = static_cast<uint8_t>(i);
        for (int b = 0; b < 8; ++b) {
            crc = (crc & 1u) ? static_cast<uint8_t>((crc >> 1) ^ 0x8Cu)
                             : static_cast<uint8_t>(crc >> 1);
        }
        out[i] = crc;
    }
}

uint8_t referenceCrc8(const uint8_t* data, size_t len) {
    uint8_t table[256];
    buildCrc8Table(table);
    uint8_t crc = 0xFF;
    for (size_t i = 0; i < len; ++i) crc = table[crc ^ data[i]];
    return crc;
}

/// 独立参考实现：与 STM32 HAL 兼容的 CRC32，逐位展开、尾部单独补零。
uint32_t referenceCrc32(const uint8_t* data, size_t len) {
    const uint32_t poly = 0x04C11DB7u;
    uint32_t crc = 0xFFFFFFFFu;
    size_t i = 0;
    while (i < len) {
        uint32_t word = 0;
        for (int k = 0; k < 4; ++k) {
            const uint8_t byte = (i < len) ? data[i] : 0;
            word |= static_cast<uint32_t>(byte) << (8 * k);
            ++i;
        }
        crc ^= word;
        for (int b = 0; b < 32; ++b) {
            crc = (crc & 0x80000000u) ? ((crc << 1) ^ poly) : (crc << 1);
        }
    }
    return crc;
}

/// 与 SerialProtocol::findAvailableSerialPorts 同源，用于决定是否触硬件。
bool anyTtyAcmPresent() {
    return !tcbs::com::McuCommunication::findAvailableSerialPorts().empty();
}

}  // namespace

int main() {
    using namespace tcbs::com;

    // ── 1. 布局与线格式 ──
    std::printf("[1] 协议结构体布局 / 前导 / CRC 覆盖范围\n");
    check(sizeof(mcu::SendPacket) == 41, "sizeof(mcu::SendPacket) == 41",
          "actual=" + std::to_string(sizeof(mcu::SendPacket)));
    check(sizeof(mcu::ReceivePacket) == 47, "sizeof(mcu::ReceivePacket) == 47",
          "actual=" + std::to_string(sizeof(mcu::ReceivePacket)));
    check(sizeof(imu::SendPacket) == 8, "sizeof(imu::SendPacket) == 8",
          "actual=" + std::to_string(sizeof(imu::SendPacket)));
    check(sizeof(imu::ReceivePacket) == 60, "sizeof(imu::ReceivePacket) == 60",
          "actual=" + std::to_string(sizeof(imu::ReceivePacket)));

    {
        mcu::SendPacket p{};
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(&p);
        check(raw[0] == 0x42 && raw[1] == 0x52 && raw[2] == 0x03,
              "MCU 发送前导 = 42 52 03");
        check(raw[3] == 36, "MCU 发送 data_size == 36",
              "actual=" + std::to_string(raw[3]));

        // 镜像 SerialProtocol::sendData 的 CRC 覆盖范围：前 sizeof-1 字节
        p.yaw_big_target_angle = -0.75;
        p.yaw_small_torque = 1.25f;
        p.crc8 = CRC8_Check_Sum(raw, sizeof(p) - 1);
        check(p.crc8 == referenceCrc8(raw, sizeof(p) - 1),
              "MCU 帧 CRC8 覆盖前 40 字节且与参考一致",
              "crc=0x" + std::to_string(p.crc8));
        check(offsetof(mcu::SendPacket, crc8) == 40, "SendPacket::crc8 偏移 == 40");
    }
    {
        mcu::ReceivePacket p{};
        p.frame_header1 = 0x42; p.frame_header2 = 0x52; p.protocol_version = 0x03;
        p.data_size = 42;
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(&p);
        check(raw[0] == 0x42 && raw[1] == 0x52 && raw[2] == 0x03 && raw[3] == 42,
              "MCU 接收前导/长度 = 42 52 03 42");
        check(offsetof(mcu::ReceivePacket, yaw_big_angle) == 12 &&
                  offsetof(mcu::ReceivePacket, yaw_small_angle) == 24 &&
                  offsetof(mcu::ReceivePacket, chassis_imu_yaw) == 32 &&
                  offsetof(mcu::ReceivePacket, mcu2_seq) == 45 &&
                  offsetof(mcu::ReceivePacket, crc8) == 46,
              "MCU 接收关键字段偏移与电控约定一致");
    }
    {
        imu::ReceivePacket p{};
        p.frame_header1 = 0xA7; p.frame_header2 = 0xB6; p.frame_header3 = 0xC5;
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(&p);
        check(raw[0] == 0xA7 && raw[1] == 0xB6 && raw[2] == 0xC5,
              "IMU 接收前导 = A7 B6 C5");
        // 镜像 SerialProtocol::sendData：CRC32 覆盖前 sizeof-4 字节，写在帧尾
        check(offsetof(imu::ReceivePacket, crc32) == sizeof(imu::ReceivePacket) - 4,
              "imu::ReceivePacket::crc32 位于帧尾");
        p.crc32 = CRC32_Calculate(raw, sizeof(p) - sizeof(uint32_t));
        check(p.crc32 == referenceCrc32(raw, sizeof(p) - sizeof(uint32_t)),
              "IMU 帧 CRC32 与参考实现一致");
    }

    // ── 2. CRC8 ──
    std::printf("[2] CRC8\n");
    {
        uint8_t ref_table[256];
        buildCrc8Table(ref_table);
        bool same = true;
        for (int i = 0; i < 256; ++i)
            if (ref_table[i] != CRC8_TAB[i]) same = false;
        check(same, "CRC8_TAB 256 项与独立生成表逐项一致");
        check(CRC8_Check_Sum(nullptr, 0) == 0xFF, "空指针返回 0xFF");

        Rng rng;
        bool func_same = true;
        for (int trial = 0; trial < 200 && func_same; ++trial) {
            std::vector<uint8_t> buf(1 + (rng.nextByte() % 64));
            for (auto& b : buf) b = rng.nextByte();
            if (CRC8_Check_Sum(buf.data(), buf.size()) !=
                referenceCrc8(buf.data(), buf.size()))
                func_same = false;
        }
        check(func_same, "CRC8_Check_Sum 与参考实现 200 组随机输入一致");
    }

    // ── 3. CRC32 ──
    std::printf("[3] CRC32\n");
    {
        check(CRC32_Calculate(nullptr, 0) == 0xFFFFFFFFu, "空输入 == 0xFFFFFFFF");
        Rng rng;
        bool same = true;
        for (int trial = 0; trial < 200 && same; ++trial) {
            std::vector<uint8_t> buf(rng.nextByte() % 97);  // 覆盖 4 的倍数与非倍数
            for (auto& b : buf) b = rng.nextByte();
            if (CRC32_Calculate(buf.data(), buf.size()) !=
                referenceCrc32(buf.data(), buf.size()))
                same = false;
        }
        check(same, "CRC32_Calculate 与参考实现 200 组随机输入一致");
    }

    // ── 4. McuDataPreprocessor 映射 ──
    std::printf("[4] McuDataPreprocessor 映射\n");
    {
        const McuDataPreprocessor pre;  // 默认（已标定）参数

        // pitch 两段各自线性，且互不为逆（这是设计意图，不是 bug）
        mcu::SendPacket s{};
        s.pitch_target_angle = 0.5f;
        const mcu::SendPacket sp = pre.processSend(s);
        const double exp_pitch_send = 21.337421 * 0.5 + (-6.708668);
        check(std::fabs(sp.pitch_target_angle - exp_pitch_send) < 1e-6,
              "pitch 发送映射 = 21.337421*x - 6.708668",
              "got=" + std::to_string(sp.pitch_target_angle));

        mcu::ReceivePacket r{};
        r.pitch_angle = 32768.0f;
        const mcu::ReceivePacket rp = pre.processReceive(r);
        const double exp_pitch_recv = 0.006060 * 32768.0 - 198.645875;
        check(std::fabs(rp.pitch_angle - exp_pitch_recv) < 1e-4,
              "pitch 接收映射 = 0.006060*raw - 198.645875",
              "got=" + std::to_string(rp.pitch_angle));

        // 大 yaw：收发互为逆映射（整体镜像，scale = -1）
        mcu::SendPacket bs{};
        const double theta = 1.234;
        bs.yaw_big_target_angle = theta;
        bs.yaw_big_target_velocity = 0.5f;
        bs.yaw_big_torque = -2.0f;
        const mcu::SendPacket bsp = pre.processSend(bs);
        check(std::fabs(bsp.yaw_big_target_angle + theta) < 1e-12 &&
                  bsp.yaw_big_target_velocity == -0.5f && bsp.yaw_big_torque == 2.0f,
              "大 yaw 发送整体取负（位置/速度/力矩）");
        mcu::ReceivePacket br{};
        br.yaw_big_angle = bsp.yaw_big_target_angle;
        br.yaw_big_omega = bsp.yaw_big_target_velocity;
        const mcu::ReceivePacket brp = pre.processReceive(br);
        check(std::fabs(brp.yaw_big_angle - theta) < 1e-12,
              "大 yaw 接收映射为发送的逆（平衡判据）",
              "got=" + std::to_string(brp.yaw_big_angle));

        // 小 yaw：零位标定 offset，且「下发值 == 编码器值 ⇒ 不动」成立
        mcu::SendPacket ss{};
        ss.yaw_small_target_angle = 0.0f;
        const mcu::SendPacket ssp = pre.processSend(ss);
        check(std::fabs(ssp.yaw_small_target_angle - 1.025466) < 1e-6,
              "小 yaw 关节角 0 → 下发 +1.025466（电控零位）",
              "got=" + std::to_string(ssp.yaw_small_target_angle));
        mcu::ReceivePacket sr{};
        sr.yaw_small_angle = 1.025466f;
        const mcu::ReceivePacket srp = pre.processReceive(sr);
        check(std::fabs(srp.yaw_small_angle) < 1e-6,
              "小 yaw 电控回读 +1.025466 → 关节角 0",
              "got=" + std::to_string(srp.yaw_small_angle));
        mcu::ReceivePacket sr2{};
        sr2.yaw_small_angle = ssp.yaw_small_target_angle;
        check(std::fabs(pre.processReceive(sr2).yaw_small_angle - 0.0f) < 1e-6,
              "小 yaw 平衡：下发值回读 ⇒ 关节角不变");

        // chassis_imu_omega 符号翻转
        mcu::ReceivePacket cr{};
        cr.chassis_imu_omega = 0.75f;
        check(pre.processReceive(cr).chassis_imu_omega == -0.75f,
              "chassis_imu_omega 接收取负");

        // 不参与映射的字段原样透传
        mcu::ReceivePacket tr{};
        tr.bullet_velocity = 27.5f;
        tr.mark = 7;
        tr.color = 1;
        tr.yaw_big_temperature = 42;
        tr.yaw_small_temperature = 43;
        tr.mcu2_seq = 200;
        const mcu::ReceivePacket trp = pre.processReceive(tr);
        check(trp.bullet_velocity == 27.5f && trp.mark == 7 && trp.color == 1 &&
                  trp.yaw_big_temperature == 42 && trp.yaw_small_temperature == 43 &&
                  trp.mcu2_seq == 200,
              "温度/模式/序号等未映射字段原样透传");

        // 模式位与 auto_aim 等发送字段不被映射改写
        mcu::SendPacket ms{};
        ms.yaw_big_mode = mcu::YAW_MODE_TORQUE_PLUS_PID;
        ms.yaw_small_mode = mcu::YAW_MODE_TORQUE_ONLY;
        ms.auto_aim_enable = 1;
        ms.fire = 1;
        const mcu::SendPacket msp = pre.processSend(ms);
        check(msp.yaw_big_mode == 2 && msp.yaw_small_mode == 1 &&
                  msp.auto_aim_enable == 1 && msp.fire == 1,
              "模式位 / auto_aim / fire 不被映射改写");
    }

    // ── 5. 模板类型实例化（无硬件时才真正构造）──
    std::printf("[5] 模板类型实例化\n");
    check(!anyTtyAcmPresent(), "本机无 /dev/ttyACM*（有则跳过构造，避免占用真实串口）");
    if (!anyTtyAcmPresent()) {
        {
            McuCommunication mcu([](const mcu::ReceivePacket&) {}, false);
            check(!mcu.isOpen() && mcu.portName().empty(),
                  "McuCommunication 可构造且未打开端口");
            mcu::SendPacket out{};
            check(!mcu.sendData(out), "端口未打开时 sendData 返回 false");
        }
        {
            ImuCommunication imu([](const imu::ReceivePacket&) {}, false);
            imu::SendPacket beat{};
            check(!imu.sendData(beat), "ImuCommunication 可构造且 sendData 返回 false");
        }
        {
            RobotCommunication robot;  // 默认映射参数
            const RobotCommunication::LatestData d = robot.getLatestData();
            check(!d.mcu_valid && !d.imu_valid && d.mcu2_seq == 0,
                  "RobotCommunication 可构造，无数据时 valid 均为 false");
            const McuDataPreprocessor::LinearParams& p = robot.preprocessor().params();
            check(p.recv_small_yaw_offset == -1.025466, "默认映射参数已注入 preprocessor");
            robot.stop();
        }
    }

    std::printf("\n%s（失败 %d 项）\n", g_fail == 0 ? "全部通过" : "存在失败", g_fail);
    return g_fail == 0 ? 0 : 1;
}

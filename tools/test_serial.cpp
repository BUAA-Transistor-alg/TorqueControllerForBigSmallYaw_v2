// tools/test_serial.cpp — MCU 串口链路自检（双级 yaw 版本）
//
// 移植自 /home/huhu233/rm2027/TorqueController/src/test_serial.cpp（tcs 单轴版）。
// 与原版的差异（仅"形态"差异，程序结构保持一致）：
//   * 命名空间 tcs → tcbs，具体通信类型位于 tcbs::com
//     （com::McuCommunication / com::ImuCommunication，见 include/com/Communications.hpp）；
//   * 协议改为双级 yaw：发送包的 yaw 通道用 yaw_big_* / yaw_small_*，模式位用
//     com::mcu::YawMode 枚举（本版协议 **1 = 仅力矩**、2 = 力矩 + 内环，其余非法）；
//     接收包同时打印两个 yaw 的角度/角速度与两个电机温度；
//   * 额外打印实际选中的串口（SerialProtocol::isOpen() / portName()）——
//     include/com/SerialProtocol.hpp 里已注明这两个只读接口就是给本程序排查
//     "到底选中/打开了哪个口"用的。
//
// 运行: ./build/tcbs_test_serial，Ctrl+C 退出

#include "Communications.hpp"

#include <atomic>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <thread>

namespace tcbs {
namespace {

std::atomic<bool> keep_running{true};

void signalHandler(int /*signum*/) { keep_running = false; }

void onReceive(const com::mcu::ReceivePacket& packet) {
    std::cout << "\n[Received Packet]" << std::endl;
    std::cout << "  frame_header1:      0x" << std::hex << std::uppercase
              << static_cast<int>(packet.frame_header1) << std::dec << std::endl;
    std::cout << "  frame_header2:      0x" << std::hex << std::uppercase
              << static_cast<int>(packet.frame_header2) << std::dec << std::endl;
    std::cout << "  protocol_version:   " << static_cast<int>(packet.protocol_version) << std::endl;
    std::cout << "  data_size:          " << static_cast<int>(packet.data_size) << std::endl;
    std::cout << "  bullet_velocity:    " << std::fixed << std::setprecision(2)
              << packet.bullet_velocity << " m/s" << std::endl;
    std::cout << "  pitch_angle:        " << packet.pitch_angle << std::endl;
    // ── 大 yaw（经 MCU1↔MCU2 链路上报，更新率低且间隔不规则，值可能被保持）──
    std::cout << "  yaw_big_angle:      " << packet.yaw_big_angle << std::endl;
    std::cout << "  yaw_big_omega:      " << packet.yaw_big_omega << " rad/s" << std::endl;
    // ── 小 yaw（MCU1 直控，实时可信）──
    std::cout << "  yaw_small_angle:    " << packet.yaw_small_angle << std::endl;
    std::cout << "  yaw_small_omega:    " << packet.yaw_small_omega << " rad/s" << std::endl;
    // ── 底盘 IMU ──
    std::cout << "  chassis_imu_yaw:    " << packet.chassis_imu_yaw << std::endl;
    std::cout << "  chassis_imu_omega:  " << packet.chassis_imu_omega << " rad/s" << std::endl;
    // ── 状态 ──
    std::cout << "  mark:               " << static_cast<int>(packet.mark) << std::endl;
    std::cout << "  color:              " << static_cast<int>(packet.color) << std::endl;
    std::cout << "  auto_aim_switch:    " << static_cast<int>(packet.auto_aim_switch) << std::endl;
    std::cout << "  yaw_big_temperature:   " << static_cast<int>(packet.yaw_big_temperature) << " C" << std::endl;
    std::cout << "  yaw_small_temperature: " << static_cast<int>(packet.yaw_small_temperature) << " C" << std::endl;
    std::cout << "  mcu2_seq:           " << static_cast<int>(packet.mcu2_seq) << std::endl;
    std::cout << "  crc8:               0x" << std::hex << std::uppercase
              << static_cast<int>(packet.crc8) << std::dec << std::endl;
    std::cout << std::endl;
}

}  // namespace
}  // namespace tcbs

// main() 必须留在全局命名空间（否则不是程序入口）；
// 下面把 namespace tcbs 内的名字引入作用域，便于 main() 直接使用。
using namespace tcbs;

int main() {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    std::cout << "Starting Serial Communication Test (dual yaw)..." << std::endl;
    std::cout << "SendPacket size: " << sizeof(com::mcu::SendPacket) << " bytes" << std::endl;
    std::cout << "ReceivePacket size: " << sizeof(com::mcu::ReceivePacket) << " bytes" << std::endl;

    com::McuCommunication serial(onReceive);

    // 链路自检：构造即尝试选口/打开，这里把结果打出来（现场排查"为什么收不到"）
    std::cout << "Port open: " << (serial.isOpen() ? "yes" : "no")
              << ", port: " << (serial.portName().empty() ? "<none>" : serial.portName())
              << std::endl;

    std::cout << "Sending packets... (auto_aim=1, fire=0, pitch=10.0, 两轴仅力矩、零力矩)" << std::endl;
    std::cout << "Press Ctrl+C to exit." << std::endl;

    int send_count = 0;
    while (keep_running) {
        com::mcu::SendPacket packet;
        // 默认初始化的 frame_header1/2、protocol_version、data_size 已自动设置
        packet.auto_aim_enable = 1;
        packet.fire = 0;
        packet.pitch_target_angle = 10.0f;
        // ── 大 yaw：仅力矩模式、目标角/速度不参与 ──
        packet.yaw_big_mode            = com::mcu::YAW_MODE_TORQUE_ONLY;
        packet.yaw_big_target_angle    = 0.0;
        packet.yaw_big_target_velocity = 0.0f;
        packet.yaw_big_torque          = 0.0f;
        // ── 小 yaw：同理 ──
        packet.yaw_small_mode            = com::mcu::YAW_MODE_TORQUE_ONLY;
        packet.yaw_small_target_angle    = 0.0f;
        packet.yaw_small_target_velocity = 0.0f;
        packet.yaw_small_torque          = 0.0f;

        if (serial.sendData(packet)) {
            send_count++;
            if (send_count % 100 == 0) {
                std::cout << "Sent " << send_count << " packets..." << std::endl;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "\nExiting... Total packets sent: " << send_count << std::endl;

    return 0;
}

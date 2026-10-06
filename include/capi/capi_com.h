#ifndef CAPI_COM_H
#define CAPI_COM_H

#include <stddef.h>
#include <stdint.h>

/*
 * TorqueControllerForBigSmallYaw_v2 通信模块（tcbs::com::RobotCommunication）的 C 语言接口。
 *
 * 对应 C++ 头文件 include/com/Communications.hpp；C++ 内容位于命名空间
 * tcbs::com，C 符号前缀为 tcbs_com_ / TcbsCom*。
 *
 * 职责（与 C++ 侧一致）：
 *   - 建立 MCU 与 IMU 两个串口（构造即启动收发/重连线程）
 *   - MCU 收包做线性映射（McuDataPreprocessor），并把 MCU/IMU 样本喂给
 *     严格反解包构建器（FullStrictPoseBuilder）
 *   - 发送 MCU / IMU 包（MCU 发送前做线性映射）
 *
 * 生命周期：
 *     TcbsRobotComm* comm = tcbs_com_create(TCBS_IMU_ON_HEAD, NULL);
 *     if (!comm) { fprintf(stderr, "%s\n", tcbs_com_last_error()); }
 *     TcbsComLatestData data = tcbs_com_get_latest_data(comm);
 *     ...
 *     tcbs_com_destroy(comm);
 *
 * 本文件不使用任何 C++ 特性，可直接被 C 编译器包含。
 */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* IMU 安装位置（构型）：决定严格反解的运动学链                        */
/*   ON_BIG_YAW: R_world_imu = R_world_chassis · Rz(θ_b) · R_A_IMU              */
/*   ON_HEAD   : R_world_imu = R_world_chassis · Rz(θ_b) · Rz(θ_s) · Rx(θ_p) · R_H_IMU */
/* ------------------------------------------------------------------ */
typedef enum TcbsImuLocation {
    TCBS_IMU_ON_BIG_YAW = 0,
    TCBS_IMU_ON_HEAD    = 1
} TcbsImuLocation;

/* ==================================================================== */
/* MCU 数据线性映射标定参数（字段顺序与 tcbs::com::McuDataPreprocessor:: */
/* LinearParams 完全一致，默认值即当前标定值）                          */
/* ==================================================================== */
typedef struct TcbsLinearParams {
    /* ── pitch（已标定；两段互不为逆）── */
    double send_pitch_scale;
    double send_pitch_offset;
    double recv_pitch_scale;
    double recv_pitch_offset;

    /* ── 大 yaw（电控侧正方向与本工程相反 ⇒ 位置/速度/力矩收发都取负号）── */
    double recv_big_yaw_scale;
    double recv_big_yaw_offset;
    double recv_big_omega_scale;
    double send_big_yaw_scale;
    double send_big_yaw_offset;
    double send_big_velocity_scale;
    double send_big_torque_scale;

    /* ── 小 yaw（零位偏移已在收发两侧分别补齐，勿改成互逆）── */
    double recv_small_yaw_scale;
    double recv_small_yaw_offset;
    double recv_small_omega_scale;
    double send_small_yaw_scale;
    double send_small_yaw_offset;
    double send_small_velocity_scale;
    double send_small_torque_scale;
} TcbsLinearParams;

/* 返回当前标定默认值（不是全 0，而是 include/com/McuDataPreprocessor.h 里的标定常数）。 */
TcbsLinearParams tcbs_com_default_linear_params(void);

/* ==================================================================== */
/* 数据包结构体（#pragma pack(1)，与 include/com/Protocol.hpp 内存布局严格一致） */
/* ==================================================================== */
#pragma pack(push, 1)

/* ── MCU 发送包（帧长 = 3 + 1 + data_size + 1 = 41）── */
typedef struct TcbsMcuSendPacket {
    uint8_t frame_header1;          /* = 0x42 */
    uint8_t frame_header2;          /* = 0x52 */
    uint8_t protocol_version;       /* = 0x03 */
    uint8_t data_size;              /* = 36 */
    /* 通用 */
    uint8_t auto_aim_enable;        /* 自瞄总开关 */
    uint8_t fire;                   /* 火控 */
    float   pitch_target_angle;     /* pitch 目标角（由预处理器线性映射后发出） */
    /* 大 yaw */
    uint8_t yaw_big_mode;           /* 1 = 仅力矩, 2 = 力矩 + 内环, 其余非法⇒力矩 0 */
    double  yaw_big_target_angle;   /* rad，关节系，多圈连续 */
    float   yaw_big_target_velocity;/* rad/s */
    float   yaw_big_torque;         /* N·m */
    /* 小 yaw */
    uint8_t yaw_small_mode;         /* 同上 */
    float   yaw_small_target_angle; /* rad，关节系（相对大 yaw） */
    float   yaw_small_target_velocity;
    float   yaw_small_torque;       /* N·m */
    uint8_t crc8;
} TcbsMcuSendPacket;

/* ── MCU 接收包（帧长 = 3 + 1 + 42 + 1 = 47）── */
typedef struct TcbsMcuReceivePacket {
    uint8_t frame_header1;          /* = 0x42 */
    uint8_t frame_header2;          /* = 0x52 */
    uint8_t protocol_version;       /* = 0x03 */
    uint8_t data_size;              /* = 42 */
    /* 弹速 / 火控 */
    float   bullet_velocity;        /* m/s */
    float   pitch_angle;            /* pitch 关节原始角（由预处理器映射为实际角度） */
    /* 大 yaw（链路低频，另有 IMU 提供高频可信分量） */
    double  yaw_big_angle;          /* rad，多圈连续 */
    float   yaw_big_omega;          /* rad/s */
    /* 小 yaw（实时可信） */
    float   yaw_small_angle;        /* rad，相对大 yaw 的关节角 */
    float   yaw_small_omega;        /* rad/s */
    /* 底盘 IMU */
    float   chassis_imu_yaw;        /* 0 ~ 2π */
    float   chassis_imu_omega;      /* rad/s */
    /* 状态 */
    uint8_t mark;                   /* 递增标志位 */
    uint8_t color;
    uint8_t auto_aim_switch;
    uint8_t yaw_big_temperature;    /* 大 yaw 电机温度 */
    uint8_t yaw_small_temperature;  /* 小 yaw 电机温度 */
    uint8_t mcu2_seq;               /* MCU2 新样本序号（值保持时不变） */
    uint8_t crc8;
} TcbsMcuReceivePacket;

/* ── IMU 发送包（心跳，无载荷；帧长 8）── */
typedef struct TcbsImuSendPacket {
    uint8_t  frame_header1;         /* = 0xA7 */
    uint8_t  frame_header2;         /* = 0xB6 */
    uint8_t  frame_header3;         /* = 0xC5 */
    uint8_t  data_size;             /* = 0 */
    uint32_t crc32;
} TcbsImuSendPacket;

/* ── IMU 接收包 ── */
typedef struct TcbsImuReceivePacket {
    uint8_t  frame_header1;         /* = 0xA7 */
    uint8_t  frame_header2;         /* = 0xB6 */
    uint8_t  frame_header3;         /* = 0xC5 */
    uint8_t  data_size;
    float    gx;                    /* IMU 本体系角速度 (rad/s) */
    float    gy;
    float    gz;
    float    ax;                    /* 加速度 (m/s^2) */
    float    ay;
    float    az;
    double   euler_yaw;             /* 世界系欧拉角 (rad)，ZXY 约定 */
    double   euler_pitch;
    double   euler_roll;
    uint32_t dt_one_tenth_ms;       /* 本帧间隔（0.1ms 单位） */
    uint32_t crc32;
} TcbsImuReceivePacket;

#pragma pack(pop)

/* ==================================================================== */
/* 聚合数据                                                            */
/* ==================================================================== */

/*
 * 同一时刻的一致快照：
 *   imu_packet     = raw_imu_packet（IMU 不经预处理）
 *   mcu_packet     = McuDataPreprocessor 的输出（已映射）
 *   raw_mcu_packet = McuDataPreprocessor 的输入（处理前原始包）
 *   mcu2_seq       = MCU2 新样本序号
 * valid 为 0 时对应 packet 是全 0。
 */
typedef struct TcbsComLatestData {
    int                   imu_valid;
    TcbsImuReceivePacket  imu_packet;
    int                   mcu_valid;
    TcbsMcuReceivePacket  mcu_packet;
    uint8_t               mcu2_seq;
    TcbsMcuReceivePacket  raw_mcu_packet;
    TcbsImuReceivePacket  raw_imu_packet;
} TcbsComLatestData;

/*
 * 严格反解数据包（FullStrictPoseBuilder::StrictPose）。
 * 没有 valid 标志，始终可读；所需数据缺失时（从未收到该来源的样本）以 0 参与。
 * chassis_euler_* / chassis_azimuth / big_azimuth / small_azimuth / gx,gy 均由
 * 反解算法给出：底盘水平时 gx = gy = 0，倾斜时为旋转平面内的重力分量。
 *
 * ★ 角度范围约定（别混用）：
 *   chassis_euler_yaw / _pitch / _roll ：包裹值，yaw ∈ (−π, π]；
 *   chassis_azimuth / big_azimuth / small_azimuth ：**多圈连续量**（累计圈数解卷绕，
 *     不受 ±π 限制），与电控上报的多圈关节角 θ_b/θ_s 同口径 —— 下游 MPC 正是按
 *     psi_b = chassis_azimuth + θ_b 使用。需要包裹值请取 chassis_euler_yaw。
 */
typedef struct TcbsComPose {
    /* 反解输入 */
    double yaw_big_angle;        /* θ_b 关节角 */
    double yaw_small_angle;      /* θ_s 关节角（相对大 yaw） */
    double pitch_angle;          /* θ_p 关节角 */
    /* 反解结果（底盘欧拉角） */
    double chassis_euler_yaw;
    double chassis_euler_pitch;
    double chassis_euler_roll;
    /* 各环节世界方位角 */
    double chassis_azimuth;
    double big_azimuth;
    double small_azimuth;
    /* 重力在旋转平面内的分量 */
    double gx;
    double gy;
    /* 角速度 */
    double small_motor_omega;
    double small_azimuth_omega;
    double big_motor_omega;
    double big_azimuth_omega;
    double chassis_omega;
} TcbsComPose;

/* ==================================================================== */
/* 句柄与函数                                                          */
/* ==================================================================== */

/* 不透明句柄：内部是 C++ 的 tcbs::com::RobotCommunication。 */
typedef struct TcbsRobotComm TcbsRobotComm;

/*
 * 创建通信句柄（构造即启动 MCU 与 IMU 的收发/重连线程）。
 *   imu_location    IMU 安装位置
 *   linear_params   MCU 线性映射参数；传 NULL 使用默认标定值
 *                    （等价于 tcbs_com_default_linear_params()）
 * 成功返回句柄；失败返回 NULL，可通过 tcbs_com_last_error() 获取原因。
 * 必须用 tcbs_com_destroy() 释放。
 */
TcbsRobotComm* tcbs_com_create(TcbsImuLocation imu_location,
                               const TcbsLinearParams* linear_params);

/* 销毁句柄（停止并 join 串口线程）；传入 NULL 是安全的。 */
void tcbs_com_destroy(TcbsRobotComm* comm);

/* 最近一次失败的错误信息（线程局部），无错误时返回空字符串。 */
const char* tcbs_com_last_error(void);

/* 获取最新 IMU + MCU 数据的一致快照（MCU 数据已预处理）。comm 为 NULL 时返回全 0。 */
TcbsComLatestData tcbs_com_get_latest_data(TcbsRobotComm* comm);

/* 获取严格反解数据包（独立输出，始终有效）。comm 为 NULL 时返回全 0。 */
TcbsComPose tcbs_com_get_strict_pose(TcbsRobotComm* comm);

/* 发送 MCU 包（发送前按当前映射参数预处理）。成功返回 1，失败返回 0。 */
int tcbs_com_send_to_mcu(TcbsRobotComm* comm, const TcbsMcuSendPacket* packet);

/* 发送 IMU 包（无预处理）。成功返回 1，失败返回 0。 */
int tcbs_com_send_to_imu(TcbsRobotComm* comm, const TcbsImuSendPacket* packet);

/* 停止通信线程（可重复调用）。 */
void tcbs_com_stop(TcbsRobotComm* comm);

/* 就地更新 MCU 线性映射参数（无需重建句柄）。params 为 NULL 时置为默认标定值。 */
void tcbs_com_set_linear_params(TcbsRobotComm* comm, const TcbsLinearParams* params);

/* 读回当前 MCU 线性映射参数；out 不可为 NULL。 */
void tcbs_com_get_linear_params(const TcbsRobotComm* comm, TcbsLinearParams* out);

/* ==================================================================== */
/* 布局自检                                                            */
/*                                                                     */
/* 返回对应结构体的 sizeof。ctypes 侧在导入时用它核对 Python 结构体与 C  */
/* 结构体布局一致（不一致会导致跨 ABI 读取静默越界，必须早失败）。        */
/* ==================================================================== */
size_t tcbs_com_sizeof_latest_data(void);
size_t tcbs_com_sizeof_pose(void);
size_t tcbs_com_sizeof_linear_params(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CAPI_COM_H */

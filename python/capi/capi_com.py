"""libtcbs.so 的 ctypes 包装（通信模块）。

对应 C 接口：include/capi/capi_com.h，实现对 src/com 中 C++ 通信封装
``tcbs::com::RobotCommunication`` 的调用。

典型用法::

    from capi import RobotCommunication, ImuLocation, McuSendPacket, YawMode

    with RobotCommunication(ImuLocation.ON_HEAD) as comm:
        data = comm.get_latest_data()          # MCU / IMU 原始数据（MCU 已预处理）
        pose = comm.get_strict_pose()          # 严格反解数据包
        comm.send_to_mcu(McuSendPacket(
            auto_aim_enable=1, fire=0, pitch_target_angle=0.1,
            yaw_big_mode=YawMode.TORQUE_ONLY, yaw_big_target_angle=0.0, yaw_big_torque=0.5,
            yaw_small_mode=YawMode.TORQUE_ONLY, yaw_small_target_angle=0.0, yaw_small_torque=0.1,
        ))

单位：角度 rad，角速度 rad/s，力矩 N*m，时间 s。

注意：构造 ``RobotCommunication`` 即启动 MCU / IMU 两个串口的收发与重连线程。
没有硬件时会打印 "No available serial port found!" 并每 3 秒重试，这是正常行为。
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum

from .capi_dm import library_path  # noqa: F401  (对外复用同一库查找逻辑)

__all__ = [
    "ImuLocation",
    "YawMode",
    "LinearParams",
    "McuSendPacket",
    "McuReceivePacket",
    "ImuSendPacket",
    "ImuReceivePacket",
    "LatestData",
    "StrictPose",
    "RobotCommunication",
    "library_path",
]


# ---------------------------------------------------------------------------
# 与 include/capi/capi_com.h 一一对应的 C 结构体
#
# 数据包结构体必须 _pack_ = 1（与 Protocol.hpp 的 #pragma pack(1) 对应）；
# 导入时会用 tcbs_com_sizeof_* 核对总体布局，不一致立刻抛错。
# ---------------------------------------------------------------------------
class _CLinearParams(ctypes.Structure):
    _fields_ = [
        ("send_pitch_scale", ctypes.c_double),
        ("send_pitch_offset", ctypes.c_double),
        ("recv_pitch_scale", ctypes.c_double),
        ("recv_pitch_offset", ctypes.c_double),
        ("recv_big_yaw_scale", ctypes.c_double),
        ("recv_big_yaw_offset", ctypes.c_double),
        ("recv_big_omega_scale", ctypes.c_double),
        ("send_big_yaw_scale", ctypes.c_double),
        ("send_big_yaw_offset", ctypes.c_double),
        ("send_big_velocity_scale", ctypes.c_double),
        ("send_big_torque_scale", ctypes.c_double),
        ("recv_small_yaw_scale", ctypes.c_double),
        ("recv_small_yaw_offset", ctypes.c_double),
        ("recv_small_omega_scale", ctypes.c_double),
        ("send_small_yaw_scale", ctypes.c_double),
        ("send_small_yaw_offset", ctypes.c_double),
        ("send_small_velocity_scale", ctypes.c_double),
        ("send_small_torque_scale", ctypes.c_double),
    ]


class _CMcuSendPacket(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("frame_header1", ctypes.c_uint8),
        ("frame_header2", ctypes.c_uint8),
        ("protocol_version", ctypes.c_uint8),
        ("data_size", ctypes.c_uint8),
        ("auto_aim_enable", ctypes.c_uint8),
        ("fire", ctypes.c_uint8),
        ("pitch_target_angle", ctypes.c_float),
        ("yaw_big_mode", ctypes.c_uint8),
        ("yaw_big_target_angle", ctypes.c_double),
        ("yaw_big_target_velocity", ctypes.c_float),
        ("yaw_big_torque", ctypes.c_float),
        ("yaw_small_mode", ctypes.c_uint8),
        ("yaw_small_target_angle", ctypes.c_float),
        ("yaw_small_target_velocity", ctypes.c_float),
        ("yaw_small_torque", ctypes.c_float),
        ("crc8", ctypes.c_uint8),
    ]


class _CMcuReceivePacket(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("frame_header1", ctypes.c_uint8),
        ("frame_header2", ctypes.c_uint8),
        ("protocol_version", ctypes.c_uint8),
        ("data_size", ctypes.c_uint8),
        ("bullet_velocity", ctypes.c_float),
        ("pitch_angle", ctypes.c_float),
        ("yaw_big_angle", ctypes.c_double),
        ("yaw_big_omega", ctypes.c_float),
        ("yaw_small_angle", ctypes.c_float),
        ("yaw_small_omega", ctypes.c_float),
        ("chassis_imu_yaw", ctypes.c_float),
        ("chassis_imu_omega", ctypes.c_float),
        ("mark", ctypes.c_uint8),
        ("color", ctypes.c_uint8),
        ("auto_aim_switch", ctypes.c_uint8),
        ("yaw_big_temperature", ctypes.c_uint8),
        ("yaw_small_temperature", ctypes.c_uint8),
        ("mcu2_seq", ctypes.c_uint8),
        ("crc8", ctypes.c_uint8),
    ]


class _CImuSendPacket(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("frame_header1", ctypes.c_uint8),
        ("frame_header2", ctypes.c_uint8),
        ("frame_header3", ctypes.c_uint8),
        ("data_size", ctypes.c_uint8),
        ("crc32", ctypes.c_uint32),
    ]


class _CImuReceivePacket(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("frame_header1", ctypes.c_uint8),
        ("frame_header2", ctypes.c_uint8),
        ("frame_header3", ctypes.c_uint8),
        ("data_size", ctypes.c_uint8),
        ("gx", ctypes.c_float),
        ("gy", ctypes.c_float),
        ("gz", ctypes.c_float),
        ("ax", ctypes.c_float),
        ("ay", ctypes.c_float),
        ("az", ctypes.c_float),
        ("euler_yaw", ctypes.c_double),
        ("euler_pitch", ctypes.c_double),
        ("euler_roll", ctypes.c_double),
        ("dt_one_tenth_ms", ctypes.c_uint32),
        ("crc32", ctypes.c_uint32),
    ]


class _CLatestData(ctypes.Structure):
    _fields_ = [
        ("imu_valid", ctypes.c_int),
        ("imu_packet", _CImuReceivePacket),
        ("mcu_valid", ctypes.c_int),
        ("mcu_packet", _CMcuReceivePacket),
        ("mcu2_seq", ctypes.c_uint8),
        ("raw_mcu_packet", _CMcuReceivePacket),
        ("raw_imu_packet", _CImuReceivePacket),
    ]


class _CPose(ctypes.Structure):
    _fields_ = [
        ("yaw_big_angle", ctypes.c_double),
        ("yaw_small_angle", ctypes.c_double),
        ("pitch_angle", ctypes.c_double),
        ("chassis_euler_yaw", ctypes.c_double),
        ("chassis_euler_pitch", ctypes.c_double),
        ("chassis_euler_roll", ctypes.c_double),
        ("chassis_azimuth", ctypes.c_double),
        ("big_azimuth", ctypes.c_double),
        ("small_azimuth", ctypes.c_double),
        ("gx", ctypes.c_double),
        ("gy", ctypes.c_double),
        ("small_motor_omega", ctypes.c_double),
        ("small_azimuth_omega", ctypes.c_double),
        ("big_motor_omega", ctypes.c_double),
        ("big_azimuth_omega", ctypes.c_double),
        ("chassis_omega", ctypes.c_double),
    ]


# ---------------------------------------------------------------------------
# 动态库加载：在 capi_dm 已加载的库上补齐 comm 相关符号签名
# ---------------------------------------------------------------------------
def _load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(library_path())

    lib.tcbs_com_create.restype = ctypes.c_void_p
    lib.tcbs_com_create.argtypes = [ctypes.c_int, ctypes.POINTER(_CLinearParams)]

    lib.tcbs_com_destroy.restype = None
    lib.tcbs_com_destroy.argtypes = [ctypes.c_void_p]

    lib.tcbs_com_last_error.restype = ctypes.c_char_p
    lib.tcbs_com_last_error.argtypes = []

    lib.tcbs_com_get_latest_data.restype = _CLatestData
    lib.tcbs_com_get_latest_data.argtypes = [ctypes.c_void_p]

    lib.tcbs_com_get_strict_pose.restype = _CPose
    lib.tcbs_com_get_strict_pose.argtypes = [ctypes.c_void_p]

    lib.tcbs_com_send_to_mcu.restype = ctypes.c_int
    lib.tcbs_com_send_to_mcu.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CMcuSendPacket)]

    lib.tcbs_com_send_to_imu.restype = ctypes.c_int
    lib.tcbs_com_send_to_imu.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CImuSendPacket)]

    lib.tcbs_com_stop.restype = None
    lib.tcbs_com_stop.argtypes = [ctypes.c_void_p]

    lib.tcbs_com_set_linear_params.restype = None
    lib.tcbs_com_set_linear_params.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CLinearParams)]

    lib.tcbs_com_get_linear_params.restype = None
    lib.tcbs_com_get_linear_params.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CLinearParams)]

    lib.tcbs_com_default_linear_params.restype = _CLinearParams
    lib.tcbs_com_default_linear_params.argtypes = []

    lib.tcbs_com_sizeof_latest_data.restype = ctypes.c_size_t
    lib.tcbs_com_sizeof_latest_data.argtypes = []
    lib.tcbs_com_sizeof_pose.restype = ctypes.c_size_t
    lib.tcbs_com_sizeof_pose.argtypes = []
    lib.tcbs_com_sizeof_linear_params.restype = ctypes.c_size_t
    lib.tcbs_com_sizeof_linear_params.argtypes = []

    return lib


_lib = _load_library()


def _last_error() -> str:
    raw = _lib.tcbs_com_last_error()
    return raw.decode("utf-8", errors="replace") if raw else "unknown error"


# ---------------------------------------------------------------------------
# 导入时布局自检：Python 结构体必须与 C 结构体逐字节一致，
# 否则跨 ABI 读取会静默越界，必须尽早失败。
# ---------------------------------------------------------------------------
def _check_layout() -> None:
    checks = [
        ("TcbsComLatestData", ctypes.sizeof(_CLatestData), _lib.tcbs_com_sizeof_latest_data()),
        ("TcbsComPose", ctypes.sizeof(_CPose), _lib.tcbs_com_sizeof_pose()),
        ("TcbsLinearParams", ctypes.sizeof(_CLinearParams), _lib.tcbs_com_sizeof_linear_params()),
        ("TcbsMcuSendPacket", ctypes.sizeof(_CMcuSendPacket), 41),
        ("TcbsMcuReceivePacket", ctypes.sizeof(_CMcuReceivePacket), 47),
        ("TcbsImuSendPacket", ctypes.sizeof(_CImuSendPacket), 8),
        ("TcbsImuReceivePacket", ctypes.sizeof(_CImuReceivePacket), 60),
    ]
    for name, py_size, c_size in checks:
        if py_size != c_size:
            raise RuntimeError(
                f"{name} 布局不一致：Python {py_size} B != C {c_size} B（Protocol.hpp 或 "
                f"capi_com.h 改了吗？）"
            )


_check_layout()


# ---------------------------------------------------------------------------
# Python 侧枚举与数据结构
# ---------------------------------------------------------------------------
class ImuLocation(IntEnum):
    """IMU 安装位置（构型）：决定严格反解的运动学链。"""

    ON_BIG_YAW = 0  # IMU 固定在大 yaw 转子
    ON_HEAD = 1     # IMU 固定在头部（pitch 之后）


class YawMode(IntEnum):
    """yaw 关节控制模式（协议规定：1 = 仅力矩，2 = 力矩 + 内环；其余取值非法 ⇒ 力矩按 0）。"""

    TORQUE_ONLY = 1
    TORQUE_PLUS_PID = 2


@dataclass(frozen=True)
class LinearParams:
    """MCU 数据线性映射标定参数。所有字段必填，不提供默认值。

    取当前标定默认值请用 ``LinearParams.default()``（默认值在 C++ 侧
    include/com/McuDataPreprocessor.h，不在这里重复）。改单个字段::

        import dataclasses
        lp = dataclasses.replace(LinearParams.default(), recv_small_yaw_offset=-1.03)
    """

    send_pitch_scale: float
    send_pitch_offset: float
    recv_pitch_scale: float
    recv_pitch_offset: float
    recv_big_yaw_scale: float
    recv_big_yaw_offset: float
    recv_big_omega_scale: float
    send_big_yaw_scale: float
    send_big_yaw_offset: float
    send_big_velocity_scale: float
    send_big_torque_scale: float
    recv_small_yaw_scale: float
    recv_small_yaw_offset: float
    recv_small_omega_scale: float
    send_small_yaw_scale: float
    send_small_yaw_offset: float
    send_small_velocity_scale: float
    send_small_torque_scale: float

    @classmethod
    def default(cls) -> "LinearParams":
        """C++ 侧的标定默认值。"""
        return cls._from_c(_lib.tcbs_com_default_linear_params())

    def _to_c(self) -> _CLinearParams:
        return _CLinearParams(
            send_pitch_scale=self.send_pitch_scale,
            send_pitch_offset=self.send_pitch_offset,
            recv_pitch_scale=self.recv_pitch_scale,
            recv_pitch_offset=self.recv_pitch_offset,
            recv_big_yaw_scale=self.recv_big_yaw_scale,
            recv_big_yaw_offset=self.recv_big_yaw_offset,
            recv_big_omega_scale=self.recv_big_omega_scale,
            send_big_yaw_scale=self.send_big_yaw_scale,
            send_big_yaw_offset=self.send_big_yaw_offset,
            send_big_velocity_scale=self.send_big_velocity_scale,
            send_big_torque_scale=self.send_big_torque_scale,
            recv_small_yaw_scale=self.recv_small_yaw_scale,
            recv_small_yaw_offset=self.recv_small_yaw_offset,
            recv_small_omega_scale=self.recv_small_omega_scale,
            send_small_yaw_scale=self.send_small_yaw_scale,
            send_small_yaw_offset=self.send_small_yaw_offset,
            send_small_velocity_scale=self.send_small_velocity_scale,
            send_small_torque_scale=self.send_small_torque_scale,
        )

    @classmethod
    def _from_c(cls, c: _CLinearParams) -> "LinearParams":
        return cls(
            send_pitch_scale=c.send_pitch_scale,
            send_pitch_offset=c.send_pitch_offset,
            recv_pitch_scale=c.recv_pitch_scale,
            recv_pitch_offset=c.recv_pitch_offset,
            recv_big_yaw_scale=c.recv_big_yaw_scale,
            recv_big_yaw_offset=c.recv_big_yaw_offset,
            recv_big_omega_scale=c.recv_big_omega_scale,
            send_big_yaw_scale=c.send_big_yaw_scale,
            send_big_yaw_offset=c.send_big_yaw_offset,
            send_big_velocity_scale=c.send_big_velocity_scale,
            send_big_torque_scale=c.send_big_torque_scale,
            recv_small_yaw_scale=c.recv_small_yaw_scale,
            recv_small_yaw_offset=c.recv_small_yaw_offset,
            recv_small_omega_scale=c.recv_small_omega_scale,
            send_small_yaw_scale=c.send_small_yaw_scale,
            send_small_yaw_offset=c.send_small_yaw_offset,
            send_small_velocity_scale=c.send_small_velocity_scale,
            send_small_torque_scale=c.send_small_torque_scale,
        )


# 发送包用**可变** dataclass：控制循环里逐帧改两个目标角最方便；
# 帧头 / 协议版本 / data_size 已按协议预置，crc 由底层串口发送时填充。
@dataclass
class McuSendPacket:
    """MCU 发送包（帧长 41 B）。字段语义见 include/com/Protocol.hpp。"""

    frame_header1: int = 0x42
    frame_header2: int = 0x52
    protocol_version: int = 0x03
    data_size: int = 36
    auto_aim_enable: int = 0
    fire: int = 0
    pitch_target_angle: float = 0.0
    yaw_big_mode: int = int(YawMode.TORQUE_ONLY)
    yaw_big_target_angle: float = 0.0
    yaw_big_target_velocity: float = 0.0
    yaw_big_torque: float = 0.0
    yaw_small_mode: int = int(YawMode.TORQUE_ONLY)
    yaw_small_target_angle: float = 0.0
    yaw_small_target_velocity: float = 0.0
    yaw_small_torque: float = 0.0
    crc8: int = 0

    def _to_c(self) -> _CMcuSendPacket:
        return _CMcuSendPacket(
            frame_header1=self.frame_header1,
            frame_header2=self.frame_header2,
            protocol_version=self.protocol_version,
            data_size=self.data_size,
            auto_aim_enable=self.auto_aim_enable,
            fire=self.fire,
            pitch_target_angle=self.pitch_target_angle,
            yaw_big_mode=self.yaw_big_mode,
            yaw_big_target_angle=self.yaw_big_target_angle,
            yaw_big_target_velocity=self.yaw_big_target_velocity,
            yaw_big_torque=self.yaw_big_torque,
            yaw_small_mode=self.yaw_small_mode,
            yaw_small_target_angle=self.yaw_small_target_angle,
            yaw_small_target_velocity=self.yaw_small_target_velocity,
            yaw_small_torque=self.yaw_small_torque,
            crc8=self.crc8,
        )

    @classmethod
    def _from_c(cls, c: _CMcuSendPacket) -> "McuSendPacket":
        return cls(
            frame_header1=c.frame_header1,
            frame_header2=c.frame_header2,
            protocol_version=c.protocol_version,
            data_size=c.data_size,
            auto_aim_enable=c.auto_aim_enable,
            fire=c.fire,
            pitch_target_angle=c.pitch_target_angle,
            yaw_big_mode=c.yaw_big_mode,
            yaw_big_target_angle=c.yaw_big_target_angle,
            yaw_big_target_velocity=c.yaw_big_target_velocity,
            yaw_big_torque=c.yaw_big_torque,
            yaw_small_mode=c.yaw_small_mode,
            yaw_small_target_angle=c.yaw_small_target_angle,
            yaw_small_target_velocity=c.yaw_small_target_velocity,
            yaw_small_torque=c.yaw_small_torque,
            crc8=c.crc8,
        )


@dataclass
class ImuSendPacket:
    """IMU 发送包（心跳，无载荷；帧长 8 B）。crc32 由底层发送时填充。"""

    frame_header1: int = 0xA7
    frame_header2: int = 0xB6
    frame_header3: int = 0xC5
    data_size: int = 0
    crc32: int = 0

    def _to_c(self) -> _CImuSendPacket:
        return _CImuSendPacket(
            frame_header1=self.frame_header1,
            frame_header2=self.frame_header2,
            frame_header3=self.frame_header3,
            data_size=self.data_size,
            crc32=self.crc32,
        )


@dataclass(frozen=True)
class McuReceivePacket:
    """MCU 接收包（帧长 47 B）。``LatestData.mcu_packet`` 是**预处理后**（已映射）的包，
    ``LatestData.raw_mcu_packet`` 是处理前的原始包。"""

    frame_header1: int
    frame_header2: int
    protocol_version: int
    data_size: int
    bullet_velocity: float
    pitch_angle: float
    yaw_big_angle: float
    yaw_big_omega: float
    yaw_small_angle: float
    yaw_small_omega: float
    chassis_imu_yaw: float
    chassis_imu_omega: float
    mark: int
    color: int
    auto_aim_switch: int
    yaw_big_temperature: int
    yaw_small_temperature: int
    mcu2_seq: int
    crc8: int

    @classmethod
    def _from_c(cls, c: _CMcuReceivePacket) -> "McuReceivePacket":
        return cls(
            frame_header1=c.frame_header1,
            frame_header2=c.frame_header2,
            protocol_version=c.protocol_version,
            data_size=c.data_size,
            bullet_velocity=c.bullet_velocity,
            pitch_angle=c.pitch_angle,
            yaw_big_angle=c.yaw_big_angle,
            yaw_big_omega=c.yaw_big_omega,
            yaw_small_angle=c.yaw_small_angle,
            yaw_small_omega=c.yaw_small_omega,
            chassis_imu_yaw=c.chassis_imu_yaw,
            chassis_imu_omega=c.chassis_imu_omega,
            mark=c.mark,
            color=c.color,
            auto_aim_switch=c.auto_aim_switch,
            yaw_big_temperature=c.yaw_big_temperature,
            yaw_small_temperature=c.yaw_small_temperature,
            mcu2_seq=c.mcu2_seq,
            crc8=c.crc8,
        )


@dataclass(frozen=True)
class ImuReceivePacket:
    """IMU 接收包（IMU 不经预处理，收发包相同）。"""

    frame_header1: int
    frame_header2: int
    frame_header3: int
    data_size: int
    gx: float
    gy: float
    gz: float
    ax: float
    ay: float
    az: float
    euler_yaw: float
    euler_pitch: float
    euler_roll: float
    dt_one_tenth_ms: int
    crc32: int

    @classmethod
    def _from_c(cls, c: _CImuReceivePacket) -> "ImuReceivePacket":
        return cls(
            frame_header1=c.frame_header1,
            frame_header2=c.frame_header2,
            frame_header3=c.frame_header3,
            data_size=c.data_size,
            gx=c.gx,
            gy=c.gy,
            gz=c.gz,
            ax=c.ax,
            ay=c.ay,
            az=c.az,
            euler_yaw=c.euler_yaw,
            euler_pitch=c.euler_pitch,
            euler_roll=c.euler_roll,
            dt_one_tenth_ms=c.dt_one_tenth_ms,
            crc32=c.crc32,
        )


@dataclass(frozen=True)
class LatestData:
    """同一时刻的一致快照。``valid`` 为 False 时对应 packet 是全 0。"""

    imu_valid: bool
    imu_packet: ImuReceivePacket
    mcu_valid: bool
    mcu_packet: McuReceivePacket
    mcu2_seq: int
    raw_mcu_packet: McuReceivePacket
    raw_imu_packet: ImuReceivePacket

    @classmethod
    def _from_c(cls, c: _CLatestData) -> "LatestData":
        return cls(
            imu_valid=bool(c.imu_valid),
            imu_packet=ImuReceivePacket._from_c(c.imu_packet),
            mcu_valid=bool(c.mcu_valid),
            mcu_packet=McuReceivePacket._from_c(c.mcu_packet),
            mcu2_seq=c.mcu2_seq,
            raw_mcu_packet=McuReceivePacket._from_c(c.raw_mcu_packet),
            raw_imu_packet=ImuReceivePacket._from_c(c.raw_imu_packet),
        )


@dataclass(frozen=True)
class StrictPose:
    """严格反解数据包（独立输出）。没有 valid 标志，始终可读；缺失数据以 0 参与。

    注意：反解算法（``chassis_euler_*`` / ``*_azimuth`` / ``gx``, ``gy``）尚未实现，
    相关字段目前为 0；``yaw_big_angle`` / ``yaw_small_angle`` / ``pitch_angle`` 来自 MCU。
    """

    yaw_big_angle: float
    yaw_small_angle: float
    pitch_angle: float
    chassis_euler_yaw: float
    chassis_euler_pitch: float
    chassis_euler_roll: float
    chassis_azimuth: float
    big_azimuth: float
    small_azimuth: float
    gx: float
    gy: float
    small_motor_omega: float
    small_azimuth_omega: float
    big_motor_omega: float
    big_azimuth_omega: float
    chassis_omega: float

    @classmethod
    def _from_c(cls, c: _CPose) -> "StrictPose":
        return cls(
            yaw_big_angle=c.yaw_big_angle,
            yaw_small_angle=c.yaw_small_angle,
            pitch_angle=c.pitch_angle,
            chassis_euler_yaw=c.chassis_euler_yaw,
            chassis_euler_pitch=c.chassis_euler_pitch,
            chassis_euler_roll=c.chassis_euler_roll,
            chassis_azimuth=c.chassis_azimuth,
            big_azimuth=c.big_azimuth,
            small_azimuth=c.small_azimuth,
            gx=c.gx,
            gy=c.gy,
            small_motor_omega=c.small_motor_omega,
            small_azimuth_omega=c.small_azimuth_omega,
            big_motor_omega=c.big_motor_omega,
            big_azimuth_omega=c.big_azimuth_omega,
            chassis_omega=c.chassis_omega,
        )


# ---------------------------------------------------------------------------
# 通信句柄
# ---------------------------------------------------------------------------
class RobotCommunication:
    """C++ ``tcbs::com::RobotCommunication`` 的 Python 句柄。

    构造即启动 MCU / IMU 两个串口的收发与重连线程；无硬件时会打印
    "No available serial port found!" 并每 3 秒重试（正常行为）。

    :param imu_location:  IMU 安装位置（构型），默认 ``ImuLocation.ON_HEAD``
    :param linear_params: MCU 线性映射参数；None 表示使用 C++ 侧标定默认值
                          （等价于 ``LinearParams.default()``）
    """

    def __init__(
        self,
        imu_location: ImuLocation = ImuLocation.ON_HEAD,
        linear_params: LinearParams | None = None,
    ):
        c_lp = None
        c_lp_ptr = None
        if linear_params is not None:
            c_lp = linear_params._to_c()
            c_lp_ptr = ctypes.byref(c_lp)
        handle = _lib.tcbs_com_create(int(imu_location), c_lp_ptr)
        if not handle:
            raise RuntimeError(f"tcbs_com_create 失败: {_last_error()}")
        self._handle = handle
        self._imu_location = ImuLocation(imu_location)

    # -- 生命周期 ---------------------------------------------------------
    def close(self) -> None:
        """销毁句柄（内部停止并 join 串口线程）。"""
        if getattr(self, "_handle", None):
            _lib.tcbs_com_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "RobotCommunication":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
        self.close()

    def _require_handle(self) -> ctypes.c_void_p:
        if not getattr(self, "_handle", None):
            raise RuntimeError("RobotCommunication 已关闭")
        return self._handle

    # -- 读取 -------------------------------------------------------------
    def get_latest_data(self) -> LatestData:
        """最新 IMU + MCU 数据的一致快照（MCU 数据已预处理）。"""
        return LatestData._from_c(_lib.tcbs_com_get_latest_data(self._require_handle()))

    def get_strict_pose(self) -> StrictPose:
        """严格反解数据包（独立输出，始终有效）。"""
        return StrictPose._from_c(_lib.tcbs_com_get_strict_pose(self._require_handle()))

    # -- 发送 -------------------------------------------------------------
    def send_to_mcu(self, packet: McuSendPacket) -> bool:
        """发送 MCU 包（底层按当前映射参数预处理后发出，CRC 由串口层填充）。"""
        return bool(
            _lib.tcbs_com_send_to_mcu(self._require_handle(), ctypes.byref(packet._to_c()))
        )

    def send_to_imu(self, packet: ImuSendPacket) -> bool:
        """发送 IMU 包（无预处理；CRC32 由串口层填充）。"""
        return bool(
            _lib.tcbs_com_send_to_imu(self._require_handle(), ctypes.byref(packet._to_c()))
        )

    def stop(self) -> None:
        """停止通信线程（可重复调用；句柄仍可读取最后一份快照）。"""
        _lib.tcbs_com_stop(self._require_handle())

    # -- 映射参数 ---------------------------------------------------------
    @property
    def linear_params(self) -> LinearParams:
        """当前 MCU 线性映射参数。"""
        out = _CLinearParams()
        _lib.tcbs_com_get_linear_params(self._require_handle(), ctypes.byref(out))
        return LinearParams._from_c(out)

    @linear_params.setter
    def linear_params(self, params: LinearParams | None) -> None:
        """就地更新映射参数（None 表示恢复 C++ 侧标定默认值）。"""
        c_lp = params._to_c() if params is not None else None
        _lib.tcbs_com_set_linear_params(
            self._require_handle(), ctypes.byref(c_lp) if c_lp is not None else None
        )

    @property
    def imu_location(self) -> ImuLocation:
        return self._imu_location

    def __repr__(self) -> str:
        state = "closed" if not getattr(self, "_handle", None) else "open"
        return f"RobotCommunication(imu_location={self._imu_location.name}, {state})"

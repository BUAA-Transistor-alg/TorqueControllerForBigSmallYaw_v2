"""二阶摆（big/small yaw）实时仿真演示。

动力学由 build/libtcbs.so 提供（见 python/capi/capi_dm.py 的 ctypes 包装），
本脚本负责：设置仿真参数、读取按键力矩、用 pygame 绘制二阶摆。

坐标系约定：向右为 x 正方向，向上为 y 正方向（屏幕 y 轴已翻转）。

基座偏航角 theta_c 由本脚本自己维护：顶部常量给出初始 THETA_C_0 / DTHETA_C_0 与
常值角加速度 DDTHETA_C，主循环每推进一步就按与仿真器子步内相同的等角加速度公式
把 (theta_c, dtheta_c) 推进一个 SIM_DT，再作为下一步的输入传给仿真器。

按键：
    A / S   关节 b 力矩 +TORQUE_B / -TORQUE_B
    K / L   关节 s 力矩 +TORQUE_S / -TORQUE_S
    SPACE   暂停 / 继续
    R       复位到初始状态
    ESC     退出

所有参数都在本文件顶部的常量区写死，不接受任何命令行参数。
"""

from __future__ import annotations

import math

import pygame

from capi import Params, Simulator, State

# ===========================================================================
# 1. 仿真参数（全部写死为常量）
# ===========================================================================

# --- 单步时间与细化倍数 -----------------------------------------------------
SIM_DT = 1.0e-3        # 仿真器单步 dt [s]
REFINEMENT = 4         # 每个 dt 内的 RK4 子步数（细化倍数）

# --- 系统物理参数（无默认值，全部显式给出） ---------------------------------
PARAMS = Params(
    # 连杆 b
    mb=1.5,            # 质量 [kg]
    Ib=0.030,          # 绕质心转动惯量 [kg*m^2]
    Pbx=0.0,         # 质心局部坐标 x [m]
    Pby=0.180,           # 质心局部坐标 y [m]
    # 连杆 s
    ms=0.40,           # 质量 [kg]
    Is=0.020,          # 绕质心转动惯量 [kg*m^2]
    Psx=0.0,         # 质心局部坐标 x [m]
    Psy=0.1,           # 质心局部坐标 y [m]
    # 关节偏移与重力场
    Dx=0.0,          # 关节 s 相对关节 b 的偏移 x [m]
    Dy=0.3,            # 关节 s 相对关节 b 的偏移 y [m]
    gx=0.0,            # 重力场加速度 x [m/s^2]
    gy=-9.81,          # 重力场加速度 y [m/s^2]（y 向上，故重力为负）
    # 摩擦
    fbc=0.020,         # 关节 b 库仑摩擦
    fbv=0.050,         # 关节 b 粘滞摩擦
    fsc=0.010,         # 关节 s 库仑摩擦
    fsv=0.020,         # 关节 s 粘滞摩擦
    lambda_=100.0,     # 平滑摩擦参数
)

# --- 初始状态：连杆 b 竖直向下，连杆 s 与之共线，静止 ------------------------
INITIAL_STATE = State(
    theta_b=math.pi,
    dtheta_b=0.0,
    theta_s=0.0,
    dtheta_s=0.0,
)

# --- 按键力矩 ---------------------------------------------------------------
TORQUE_B = 1.0         # A / S 键给出的关节 b 力矩幅值 [N*m]（范围 [-1, +1]）
TORQUE_S = 1.0         # K / L 键给出的关节 s 力矩幅值 [N*m]（范围 [-1, +1]）

# --- 外部输入 theta_c（基座偏航角）：初始状态 + 常值角加速度 -----------------
# 仿真器每步只接收本步起始时刻的 (theta_c, dtheta_c, ddtheta_c)，步内按等角加速度
# 外推；本步结束后由本脚本按同一公式把 c 状态推进到下一步（见 advance_theta_c）。
THETA_C_0 = 0.0        # 初始基座角度 [rad]
DTHETA_C_0 = 0.0       # 初始基座角速度 [rad/s]
DDTHETA_C = 0.0        # 基座角加速度 [rad/s^2]（常值；非 0 时基座持续加速旋转）

# --- 实时推进 ---------------------------------------------------------------
FPS = 60
TIME_SCALE = 1.0            # 仿真时间 / 真实时间
MAX_FRAME_SECONDS = 0.05    # 单帧最多补偿的真实时间，避免卡顿后追帧爆炸
MAX_STEPS_PER_FRAME = 2000  # 单帧最多仿真步数

# ===========================================================================
# 2. 显示参数（全部写死为常量）
# ===========================================================================
WINDOW_W = 960
WINDOW_H = 800
CAPTION = "Two-link pendulum (a/s: Tb, k/l: Ts, space: pause, r: reset, esc: quit)"

ORIGIN_PX = (560.0, 280.0)  # 关节 b 的屏幕位置（屏幕坐标 y 向下）
PX_PER_METER = 620.0        # 缩放：像素 / 米
GRID_STEP_M = 0.1           # 背景网格间距 [m]
AXIS_LEN_X_M = 0.50         # +x 轴箭头长度 [m]
AXIS_LEN_Y_M = 0.28         # +y 轴箭头长度 [m]（受窗口上边界限制）

C_BACKGROUND = (24, 26, 32)
C_GRID = (38, 42, 52)
C_AXIS_X = (200, 92, 92)
C_AXIS_Y = (108, 186, 108)
C_LINK_B = (118, 176, 255)
C_LINK_S = (255, 190, 118)
C_JOINT = (236, 236, 242)
C_COM = (255, 255, 255)
C_BASE = (150, 155, 165)
C_TEXT = (226, 229, 236)
C_TEXT_DIM = (150, 156, 168)
C_POS = (238, 96, 80)
C_NEG = (92, 152, 240)

LINK_B_WIDTH = 11
LINK_S_WIDTH = 8


# ===========================================================================
# 3. 几何与绘制
# ===========================================================================
def _rot(vx: float, vy: float, angle: float) -> tuple[float, float]:
    """二维旋转：R(angle) * (vx, vy)。"""
    c, s = math.cos(angle), math.sin(angle)
    return (c * vx - s * vy, s * vx + c * vy)


def world_to_screen(x: float, y: float) -> tuple[int, int]:
    """世界坐标（x 右、y 上，单位 m）-> 屏幕像素坐标。"""
    return (
        int(round(ORIGIN_PX[0] + x * PX_PER_METER)),
        int(round(ORIGIN_PX[1] - y * PX_PER_METER)),
    )


def compute_geometry(state: State, theta_c: float) -> dict:
    """由基座角度与两个广义坐标算出各关节点、质心与连杆端点的世界坐标。

    运动学：
        psi_b = theta_c + theta_b
        psi_s = theta_c + theta_b + theta_s
        关节 s = R(psi_b) * (Dx, Dy)
        连杆 b 质心 = R(psi_b) * (Pbx, Pby)
        连杆 s 质心 = 关节 s + R(psi_s) * (Psx, Psy)
        连杆 s 末端 = 关节 s + R(psi_s) * (2*Psx, 2*Psy)   （仅用于绘制）
    """
    psi_b = theta_c + state.theta_b
    psi_s = psi_b + state.theta_s

    joint = _rot(PARAMS.Dx, PARAMS.Dy, psi_b)
    com_b = _rot(PARAMS.Pbx, PARAMS.Pby, psi_b)
    com_s_rel = _rot(PARAMS.Psx, PARAMS.Psy, psi_s)
    com_s = (joint[0] + com_s_rel[0], joint[1] + com_s_rel[1])
    tip = (joint[0] + 2.0 * com_s_rel[0], joint[1] + 2.0 * com_s_rel[1])

    return {
        "origin": (0.0, 0.0),
        "joint": joint,
        "com_b": com_b,
        "com_s": com_s,
        "tip": tip,
        "psi_b": psi_b,
        "psi_s": psi_s,
    }


def draw_grid(surface: pygame.Surface) -> None:
    """以关节 b 为原点绘制等间距网格。"""
    for k in range(-12, 13):
        offset = k * GRID_STEP_M * PX_PER_METER
        x = int(round(ORIGIN_PX[0] + offset))
        if 0 <= x < WINDOW_W:
            pygame.draw.line(surface, C_GRID, (x, 0), (x, WINDOW_H), 1)
        y = int(round(ORIGIN_PX[1] + offset))
        if 0 <= y < WINDOW_H:
            pygame.draw.line(surface, C_GRID, (0, y), (WINDOW_W, y), 1)


def draw_axes(surface: pygame.Surface, font: pygame.font.Font) -> None:
    """绘制 +x（右）与 +y（上）坐标轴，明确坐标系约定。"""
    ox, oy = world_to_screen(0.0, 0.0)
    ax = world_to_screen(AXIS_LEN_X_M, 0.0)
    ay = world_to_screen(0.0, AXIS_LEN_Y_M)

    pygame.draw.line(surface, C_AXIS_X, (ox, oy), ax, 2)
    pygame.draw.polygon(surface, C_AXIS_X, [(ax[0], ax[1]),
                                            (ax[0] - 12, ax[1] - 5),
                                            (ax[0] - 12, ax[1] + 5)])
    pygame.draw.line(surface, C_AXIS_Y, (ox, oy), ay, 2)
    pygame.draw.polygon(surface, C_AXIS_Y, [(ay[0], ay[1]),
                                            (ay[0] - 5, ay[1] + 12),
                                            (ay[0] + 5, ay[1] + 12)])

    surface.blit(font.render("+x", True, C_AXIS_X), (ax[0] + 6, ax[1] - 8))
    surface.blit(font.render("+y", True, C_AXIS_Y), (ay[0] + 8, ay[1] - 16))


def draw_base(surface: pygame.Surface) -> None:
    """绘制固定基座标记。"""
    ox, oy = world_to_screen(0.0, 0.0)
    pygame.draw.circle(surface, C_BASE, (ox, oy), 9)
    pygame.draw.line(surface, C_BASE, (ox - 26, oy + 16), (ox + 26, oy + 16), 3)
    for i in range(-4, 5):
        x = ox + i * 7
        pygame.draw.line(surface, C_BASE, (x, oy + 16), (x - 8, oy + 26), 2)


def draw_pendulum(surface: pygame.Surface, state: State, theta_c: float) -> dict:
    """绘制二阶摆本体，返回几何量供 HUD 使用。"""
    geo = compute_geometry(state, theta_c)

    origin_px = world_to_screen(*geo["origin"])
    joint_px = world_to_screen(*geo["joint"])
    tip_px = world_to_screen(*geo["tip"])
    com_b_px = world_to_screen(*geo["com_b"])
    com_s_px = world_to_screen(*geo["com_s"])

    # 连杆 b：关节 b -> 关节 s
    pygame.draw.line(surface, C_LINK_B, origin_px, joint_px, LINK_B_WIDTH)
    # 连杆 s：关节 s -> 末端
    pygame.draw.line(surface, C_LINK_S, joint_px, tip_px, LINK_S_WIDTH)

    # 质心标记
    pygame.draw.circle(surface, C_COM, com_b_px, 5)
    pygame.draw.circle(surface, C_COM, com_s_px, 5)
    # 关节 s
    pygame.draw.circle(surface, C_JOINT, joint_px, 8)
    pygame.draw.circle(surface, C_BACKGROUND, joint_px, 4)
    # 末端
    pygame.draw.circle(surface, C_LINK_S, tip_px, 6)

    return geo


def draw_torque_bar(
    surface: pygame.Surface,
    x: int,
    y: int,
    torque: float,
    max_torque: float,
    label: str,
    font: pygame.font.Font,
) -> None:
    """用带符号的条形显示当前力矩。"""
    width = 160
    height = 12
    pygame.draw.rect(surface, (52, 56, 66), pygame.Rect(x, y, width, height), border_radius=3)
    frac = max(-1.0, min(1.0, torque / max_torque))
    center = x + width // 2
    half = width // 2
    bar = int(round(half * frac))
    color = C_POS if torque > 0 else (C_NEG if torque < 0 else C_TEXT_DIM)
    if bar != 0:
        rect = pygame.Rect(min(center, center + bar), y + 1, abs(bar), height - 2)
        pygame.draw.rect(surface, color, rect, border_radius=2)
    pygame.draw.line(surface, C_TEXT_DIM, (center, y - 2), (center, y + height + 2), 1)
    surface.blit(font.render(label, True, C_TEXT_DIM), (x, y + height + 4))
    surface.blit(
        font.render(f"{torque:+.2f} N*m", True, color),
        (x + width + 10, y),
    )


def render(
    surface: pygame.Surface,
    font: pygame.font.Font,
    font_big: pygame.font.Font,
    state: State,
    theta_c: float,
    dtheta_c: float,
    torque_b: float,
    torque_s: float,
    sim_time: float,
    paused: bool,
    fps: float,
) -> None:
    surface.fill(C_BACKGROUND)
    draw_grid(surface)
    draw_axes(surface, font)
    draw_base(surface)
    geo = draw_pendulum(surface, state, theta_c)

    # --- HUD 面板 ---
    x0, y0 = 30, 26
    lines = [
        f"dt = {SIM_DT:g} s    refinement = {REFINEMENT}    substep = {SIM_DT / REFINEMENT:g} s",
        f"sim time = {sim_time:8.3f} s    fps = {fps:5.1f}" + ("    [PAUSED]" if paused else ""),
        f"theta_b = {state.theta_b:+.4f} rad   dtheta_b = {state.dtheta_b:+.4f} rad/s",
        f"theta_s = {state.theta_s:+.4f} rad   dtheta_s = {state.dtheta_s:+.4f} rad/s",
        f"theta_c = {theta_c:+.4f} rad   dtheta_c = {dtheta_c:+.4f} rad/s",
        f"ddtheta_c = {DDTHETA_C:+.4f} rad/s^2",
        f"psi_b   = {geo['psi_b']:+.4f} rad   psi_s    = {geo['psi_s']:+.4f} rad",
    ]
    n_lines = len(lines)
    panel = pygame.Surface((470, n_lines * 22 + 84), pygame.SRCALPHA)
    panel.fill((12, 14, 18, 190))
    surface.blit(panel, (16, 16))

    for i, text in enumerate(lines):
        surface.blit(font.render(text, True, C_TEXT), (x0, y0 + i * 22))

    keys_y = y0 + n_lines * 22 + 4
    surface.blit(
        font_big.render("Keys: A/S -> Tb   K/L -> Ts", True, C_TEXT),
        (x0, keys_y),
    )
    surface.blit(
        font.render("SPACE pause / resume    R reset    ESC quit", True, C_TEXT_DIM),
        (x0, keys_y + 28),
    )

    # 力矩条
    bar_x, bar_y = WINDOW_W - 360, 40
    draw_torque_bar(surface, bar_x, bar_y, torque_b, TORQUE_B, "Tb (A/S)", font)
    draw_torque_bar(surface, bar_x, bar_y + 70, torque_s, TORQUE_S, "Ts (K/L)", font)


# ===========================================================================
# 4. 输入与主循环
# ===========================================================================
def advance_theta_c(
    theta_c: float, dtheta_c: float, ddtheta_c: float, dt: float
) -> tuple[float, float]:
    """把基座状态 (theta_c, dtheta_c) 推进一个 dt。

    与仿真器 RK4 子步内的假设保持一致（等角加速度外推）：
        theta_c(t + dt) = theta_c + dtheta_c*dt + 0.5*ddtheta_c*dt^2
        dtheta_c(t + dt) = dtheta_c + ddtheta_c*dt
    """
    return (
        theta_c + dtheta_c * dt + 0.5 * ddtheta_c * dt * dt,
        dtheta_c + ddtheta_c * dt,
    )


def read_torques() -> tuple[float, float]:
    """按当前按键状态给出两个驱动力矩。

    A 与 S 等大反向，K 与 L 等大反向；同时按下或都不按则为 0。
    """
    keys = pygame.key.get_pressed()

    torque_b = 0.0
    if keys[pygame.K_a]:
        torque_b += TORQUE_B
    if keys[pygame.K_s]:
        torque_b -= TORQUE_B

    torque_s = 0.0
    if keys[pygame.K_k]:
        torque_s += TORQUE_S
    if keys[pygame.K_l]:
        torque_s -= TORQUE_S

    return torque_b, torque_s


def main() -> int:
    pygame.init()
    try:
        screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    except pygame.error as exc:  # 例如没有可用显示设备
        print(f"无法创建 pygame 窗口: {exc}")
        pygame.quit()
        return 1

    pygame.display.set_caption(CAPTION)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)
    font_big = pygame.font.SysFont(None, 26)

    sim = Simulator(PARAMS, SIM_DT, REFINEMENT)
    sim.set_state(INITIAL_STATE)

    state = INITIAL_STATE
    # 脚本自己维护的基座状态
    theta_c = THETA_C_0
    dtheta_c = DTHETA_C_0
    sim_time = 0.0
    accumulator = 0.0
    paused = False
    running = True

    try:
        while running:
            frame_seconds = clock.tick(FPS) / 1000.0
            frame_seconds = min(frame_seconds, MAX_FRAME_SECONDS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        sim.set_state(INITIAL_STATE)
                        state = INITIAL_STATE
                        theta_c = THETA_C_0
                        dtheta_c = DTHETA_C_0
                        sim_time = 0.0
                        accumulator = 0.0

            torque_b, torque_s = read_torques()

            if not paused:
                accumulator += frame_seconds * TIME_SCALE
                steps = 0
                while accumulator >= SIM_DT and steps < MAX_STEPS_PER_FRAME:
                    # 传入本步起始时刻的基座状态
                    state = sim.step(torque_b, torque_s, theta_c, dtheta_c, DDTHETA_C)
                    # 本步结束后按同一公式把基座状态推进到下一步
                    theta_c, dtheta_c = advance_theta_c(theta_c, dtheta_c, DDTHETA_C, SIM_DT)
                    sim_time += SIM_DT
                    accumulator -= SIM_DT
                    steps += 1
                if steps >= MAX_STEPS_PER_FRAME:
                    accumulator = 0.0

            render(screen, font, font_big, state, theta_c, dtheta_c, torque_b, torque_s,
                   sim_time, paused, clock.get_fps())
            pygame.display.flip()
    finally:
        sim.close()
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

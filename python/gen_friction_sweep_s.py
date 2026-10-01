"""小 yaw（关节 s）匀速往返摩擦辨识采集（用 env 仿真环境接口实现）。

原理（与库内模型严格对应，见 src/dm/dynamics.cpp 的关节 s 方程）
------------------------------------------------------------------
关节 s 的方程（基座静止 θ_c=0）：
    M12·θ̈_b + M22·θ̈_s + C2 + G2 = Qs = Ts − fsv·θ̇_s − fsc·tanh(λ·θ̇_s)

让 θ_b 被固定住（θ̇_b≈θ̈_b≈0）、θ_s 以恒定 ω 匀速往返：
    * M22 = Is + ms|Ps|² 只与参数有关（与 θ_s 无关）；
    * M12·θ̈_b ≈ 0、C2 = −ms·(dh/dθ_s)·θ̇_b² ≈ 0、M22·θ̈_s ≈ 0（匀速段）；
    ⇒ 稳态关系退化成：  Ts = G2(ψ_s) + fsv·ω + fsc·tanh(λ·ω)

对**同一段以 0 为中心对称的位置窗口**取正、反两趟：
    ΔTs = Ts(+ω) − Ts(−ω) = fsv·Δθ̇_s + fsc·Δtanh(λ·θ̇_s)
G2 只与位置有关，两趟扫过同一对称区间 ⇒ 一阶项精确抵消。因此：
    * **不需要任何已辨识的动力学/重力参数**（G2 通常比摩擦大一个量级，直接扣模型会漏进去）；
    * θ_b 有小漂移也不怕（对称区间的 sin/cos 均值只通过 sin(A)/A 弱依赖幅值）。

大 yaw 能整圈旋转（整圈平均掉 G1），小 yaw 只有 ±35° 硬限位，所以拿"正反行程镜像"
换"整圈平均"——这就是本脚本对应 gen_friction_sweep.py 的设计。

怎么保证"带限位也能建立匀速"（本脚本的核心）
--------------------------------------------
建立速度是**距离**问题不是时间问题：实机反解 M22≈0.006 kg·m²、τ_max=1 N·m
⇒ a_max≈170 rad/s²，从静止拉到 ω 只要 ω²/(2a)：ω=3 rad/s 也才 1.5°。
吃行程的是"保持速度"，所以幅值按速度缩放：

    A(ω) = clip(ω·SS_CV_TIME/2 + ω²/(2·SS_A_REF), SS_AMP_MIN_DEG, 幅值上限)

即每个速度的**匀速段只占 SS_CV_TIME 秒**；高速时幅值被上限截断、匀速段变短，
就多跑几趟（每趟仍覆盖同一段位置，差分照样成立），而不是去等它"稳"。
规划时按 2·gate/(ω·dt) 预估单趟样本数，不足 SS_MIN_SAMPLES 就丢弃该速度并打印原因。

安全（绝不依赖电控在 ±35° 处的保护——那里力矩会被削弱，发出的力矩≠实际力矩）
--------------------------------------------------------------------------
    * 参考速度**限斜坡** SS_A_REF（不是从 +ω 跳到 −ω）⇒ 反作用力矩 ≈ M12·SS_A_REF
      有界，不会把抱死的大 yaw 顶飞；反向点 |θ_s| + ω²/(2·SS_A_REF) ≥ A 正是斜坡
      所需距离，实际峰值 ≲ A，永远够不到 ±35°（幅值上限已经取
      THETA_S_TARGET_DEG）；
    * 幅值上限就是 THETA_S_TARGET_DEG（±30°，不再额外缩到 28°）：主动控制范围
      朝 30° 控，换向峰值偶尔进到 30~35° 也没关系；
    * 单趟越过 THETA_S_LIMIT_DEG(±35°) 只作废**这一趟**（电控在此介入削弱力矩，
      发出的力矩≠实际力矩），该速度继续跑、其余趟照常采；
    * 每个 step 的力矩都先按 TAU_*_MAX 硬限幅。

大 yaw 固定
-----------
默认 ``Tb = 0``：完全松开大 yaw 力矩，交给人工/机械完全固定（会记录 θ̇_b、⟨θ̈_b⟩ 与
正反两趟 θ_b 均值差，超限的趟/对直接剔除）。
``--hold-big`` 则用位置环把 θ_b 抱在会话开始的角度（小 yaw 加速时反作用力矩峰值
约 0.5 N·m，会推动松开的 θ_b；主动抱死能压住它，代价是大 yaw 电机持续出力）。

"平均值"到底代不代表这一段——窗口内的波动怎么卡
------------------------------------------------
辨识用的是**窗口均值** ⟨ω⟩、⟨Ts⟩，所以卡的是**均值标准误** σ/√n，不是逐样本 std。
真机上报角速度的测量噪声底实测 ~0.026 rad/s（Sentry1 数据 >20 Hz 频段），比 tanh 的
转折尺度 1/λ=0.01 rad/s 还大：低速时逐样本 std/|ω| 看着有 490%，那全是测量噪声。
白噪声按 1/√n 平均掉，但 tanh 是凸的，E[tanh(λ(ω̄+ε))] ≠ tanh(λω̄)，这个 Jensen 偏差
正、反两趟同号、**差分抵消不掉** ⇒ 卡 λ·σ/√n ≤ 0.25（≈2% 库仑列偏差，折算 fsc 1~2%）。
真机噪声下每趟窗内样本 n≳120（≈1.2 s 匀速段）即可满足，``SS_CV_TIME=3 s`` 给的就是
这个量级；力矩侧同理卡 σ_T/√n（实测 ~1e-3 N·m）。

逐样本 std（``dtheta_s_std``、``ts_std``）照样完整记录，但只作诊断——真机噪声下它本来
就大。**相关性**波动（换向余振、大 yaw 被拖动）不随 n 下降，由另外三道判据兜住：
前半 vs 后半均值差、⟨θ̈_b⟩、θ̇_b RMS。任何一条不达标就丢这一趟，``--debug`` 打印原因。

一个反直觉的坑：**KPV 会把速度噪声直接搬进力矩**（KPV·σ_v）。KPV=1.0、σ_v=0.026 就是
0.026 N·m 的指令抖动，和摩擦本身同量级，实测真机噪声档下所有趟都被力矩判据判废；
KPV<0.3 又让换向余振衰减不完、高转速窗被余振污染（实测 ω≥1 rad/s 全灭）。默认 0.5 是
实测折中（真机 J 下 ζ≈1、τ≈23 ms）。换惯量后按 ζ≈1 重算：``KPV ≈ 2·√(KIV·M22)``。

仿真注意：默认 ``SIGMA_POS=0.01 rad``(0.57°) 是实测真机（1.1e-3 rad）的 9 倍，会把
``SS_AMP_MIN_DEG=0.5°`` 的幅值控制直接冲垮；``SIGMA_VEL=0.05`` 也大于真机的 0.026。
仿真验证请用 ``--no-noise``，或按真机噪声档 ``--sigma-vel 0.026 --sigma-pos 0.0011``。

用法::

    python3 python/gen_friction_sweep_s.py --category friction_s_simA --no-noise
    python3 python/gen_friction_sweep_s.py --category friction_s_simA \
        --sigma-vel 0.026 --sigma-pos 0.0011        # 按真机噪声档的压力测试
    python3 python/gen_friction_sweep_s.py --real --category friction_s_Sentry1
    python3 python/gen_friction_sweep_s.py --real --hold-big --category friction_s_Sentry1

开始记录前先标定可用角加速度（SS_AMAX_CAL，--no-amax-cal 关掉）
----------------------------------------------------------------
先稳定到一侧 THETA_S_TARGET_DEG，再以最大力矩冲向另一侧、到中点后反过来用最大力矩
刹停；对**加速段**（始终在 ±THETA_S_TARGET_DEG 之内）拟合 θ(t)=θ0+v0·t+½a·t²，
正反各一段算一组，来回 SS_AMAX_CYCLES 次取平均（正反平均把重力矩 G2 消掉）。
标定值有两个用途：
  1. 换向斜坡取 ``min(SS_A_REF, SS_A_REF_FRAC×实测)``，保证反向真的刹得住；
  2. 速度环的**加速度前馈** ``(dv_ref/dt)·τ_max/a_max``——不加它，换向斜坡期间纯靠
     P 项跟不上参考（误差 ≈ M_eff·a/KPV），实测 ω=3 会多跑 5° 冲到 35.5°；加了以后
     峰值 30.4°(ω=1)/32.4°(ω=3)，不再触发 >35° 作废。

输出
----
``data/<类别>/sweep_s_<YYYYMMDD_HHMMSS>.npz``（``--out a/b.npz`` → ``a/b_<时间戳>.npz``）；
缺省类别目录：仿真 ``data/friction_s/``、真机 ``data/friction_s_real/``。
辨识端读整个目录（glob ``sweep_*.npz``）并按 npz 里的 ``joint`` 标记只取小 yaw 的行，
所以同一目录里混放大小 yaw 的 sweep 也不会串。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sim_config as cfg                                    # noqa: E402
from capi import ImuLocation                                 # noqa: E402
from env import RealEnv, SimEnv                              # noqa: E402


# ===========================================================================
# 控制回路（关节 s 速度环 PI + 可选重力前馈；关节 b 可选位置环）
# ===========================================================================
class SweepLoop:
    """在一个 env 上跑：归位 → 等速往返 → 逐趟验收。状态（积分/低通）跨速度保留。"""

    def __init__(self, env, hold_big: bool, gravity_ff: bool, tau_s_max: float):
        self.env = env
        self.hold_big = bool(hold_big)
        self.gravity_ff = bool(gravity_ff)
        self.tau_s_max = float(tau_s_max)
        self._int_s = 0.0                 # θ_s 速度环积分
        self._int_b = 0.0                 # θ_b 位置环积分（--hold-big）
        self._filt: list[float] | None = None
        self.theta_b0 = float(env.state().theta_b)
        self.a_max_cal = 0.0              # calibrate_amax 测出的可用角加速度（0=没标定）

    # ---------------- 内部：采样（低通）+ 力矩 ----------------
    def _measure(self):
        """返回 (原始带噪状态, 控制用低通估计 [θ_b, θ̇_b, θ_s, θ̇_s])。"""
        st = self.env.state()
        a = cfg.SS_CTRL_LPF_ALPHA
        if self._filt is None:
            self._filt = [st.theta_b, st.dtheta_b, st.theta_s, st.dtheta_s]
        f = self._filt
        f[0] += a * (st.theta_b - f[0])
        f[1] += a * (st.dtheta_b - f[1])
        f[2] += a * (st.theta_s - f[2])
        f[3] += a * (st.dtheta_s - f[3])
        return st, f

    def _b_torque(self, f: list[float]) -> float:
        """关节 b 的力矩：默认 0（松开），--hold-big 时位置环抱死。"""
        if not self.hold_big:
            return 0.0
        g1 = 0.0
        if self.gravity_ff:
            g1, _ = self.env.gravity_torque(f[0], f[2])
        eb = self.theta_b0 - f[0]
        self._int_b = float(np.clip(self._int_b + eb * self.env.dt,
                                    -cfg.SS_INT_CLAMP, cfg.SS_INT_CLAMP))
        Tb = (g1 + cfg.SS_KPB_HOLD * eb + cfg.SS_KIB_HOLD * self._int_b
              - cfg.SS_KDB_HOLD * f[1])
        return float(np.clip(Tb, -cfg.TAU_B_MAX, cfg.TAU_B_MAX))

    def _raw(self, ts_cmd: float, f: list[float]) -> tuple[float, float]:
        """给关节 s 一个**直接**力矩指令（只叠重力前馈 G2），不经速度环。"""
        g2 = 0.0
        if self.gravity_ff:
            _, g2 = self.env.gravity_torque(f[0], f[2])
        return self._b_torque(f), float(np.clip(g2 + ts_cmd, -self.tau_s_max,
                                                self.tau_s_max))

    def _torques(self, v_ref_s: float, f: list[float],
                 a_demand: float = 0.0) -> tuple[float, float]:
        """给定 θ̇_s 参考，算出本帧 (Tb, Ts)（已硬限幅）。

        ``a_demand`` 是参考速度的斜率 [rad/s²]：换向斜坡期间速度环纯靠 P 项跟不上
        （误差 ≈ M_eff·a/KPV，ω=3 时实测多跑 5° 冲到 35.5°），所以按标定出的 a_max
        直接做**加速度前馈**：需要的力矩 = a_demand·τ_max/a_max。

        两个环都用**条件积分**抗饱和：只在力矩未饱和时累加积分。否则在 ω=±3 rad/s
        这种 P 项本身就超过力矩上限的速度点上，积分会在饱和期间狂涨，解锁后把关节
        甩到 ±10 rad/s（实测 |v1−v2|≈9，整趟报废）。
        """
        dt = self.env.dt
        g1 = g2 = 0.0
        if self.gravity_ff:
            g1, g2 = self.env.gravity_torque(f[0], f[2])
        e = v_ref_s - f[3]
        ff = a_demand * self.tau_s_max / self.a_max_cal if self.a_max_cal > 0.0 else 0.0
        Ts_raw = g2 + ff + cfg.SS_KPV * e + cfg.SS_KIV * self._int_s
        Ts = float(np.clip(Ts_raw, -self.tau_s_max, self.tau_s_max))
        if Ts == Ts_raw:
            self._int_s = float(np.clip(self._int_s + e * dt,
                                        -cfg.SS_INT_CLAMP, cfg.SS_INT_CLAMP))
        return self._b_torque(f), Ts

    # ---------------- 归位：把 θ_s 送到往返起点 ----------------
    def goto(self, theta_s_target: float) -> bool:
        """把 θ_s 送到 θ_s_target（**独立的位置 PID**，与采集用的速度环不共用参数）。

        为什么不用串级："位置P → 速度参考 → 速度环PI"展开后等效位置增益是
        Kp = KPV·SS_REPOS_KP + KIV ≈ 11 N·m/rad（5.2° 误差即顶到 ±1 N·m 限幅），
        从两个参数完全看不出来；而且它和采集共用积分状态，goto 攒的积分会漏进第一趟。
        现在 Kp/Ki/Kd 直接可读：Kp=0.6 N·m/rad ⇒ 30° 误差才 0.31 N·m，
        实机 M22≈0.0059 时 ω_n≈10 rad/s、ζ≈1.3（不过冲、不敲关节）。

        参考轨迹从**当前位置**出发、按 SS_REPOS_V_MAX 限速推进（起点误差为 0、
        力矩从 0 长起来），PID 跟的是这条参考而不是直接盯目标——否则第一帧力矩
        就是 KP·误差（30° 时 0.31 N·m 的阶跃）。

        收敛判据：位置进容差**并且速度也停了**。只判位置会在关节还以 0.7 rad/s
        运动时就返回（实测），换速度和标定都会带着残余速度起步。容差按噪声自适应。
        """
        env = self.env
        dt = env.dt
        tol = max(cfg.SS_REPOS_TOL_RAD, 3.0 * env.sigma[0] * 0.5)
        tol_v = max(cfg.SS_REPOS_TOL_VEL, 2.0 * env.sigma[1])
        integ = 0.0                                   # 归位自己的积分，不碰 _int_s
        ref = float(self._measure()[1][2])            # 参考轨迹从当前位置出发
        for _ in range(cfg.SS_REPOS_MAX_STEPS):
            _st, f = self._measure()
            if abs(theta_s_target - f[2]) < tol and abs(f[3]) < tol_v:
                return True
            ref += float(np.clip(theta_s_target - ref, -cfg.SS_REPOS_V_MAX * dt,
                                 cfg.SS_REPOS_V_MAX * dt))
            e = ref - f[2]
            g2 = 0.0
            if self.gravity_ff:
                _, g2 = env.gravity_torque(f[0], f[2])
            ts_raw = (g2 + cfg.SS_GOTO_KP * e + cfg.SS_GOTO_KI * integ
                      - cfg.SS_GOTO_KD * f[3])
            Ts = float(np.clip(ts_raw, -self.tau_s_max, self.tau_s_max))
            if Ts == ts_raw:                          # 条件积分抗饱和
                integ = float(np.clip(integ + e * dt,
                                      -cfg.SS_GOTO_I_CLAMP, cfg.SS_GOTO_I_CLAMP))
            env.step(self._b_torque(f), Ts)
        return False

    # ---------------- 最大可用角加速度标定 ----------------
    def calibrate_amax(self, cycles: int) -> dict:
        """在 ±THETA_S_TARGET_DEG 内用**最大力矩**测可用角加速度。

        做法：先稳定到一侧 THETA_S_TARGET_DEG，再以最大力矩冲向另一侧，到中点后
        反过来用最大力矩制动，停在另一侧；对**加速段**（始终在 ±THETA_S_TARGET_DEG
        之内）拟合 θ(t)=θ0+v0·t+½a·t²，来回多次取平均。
        正、反两向都做：重力矩 G2 在两向符号相反、平均后抵消，剩下
        a ≈ (τ_max − f_sc − f_sv·⟨v⟩)/M22，比 τ_max/M22 略小——作为设计上限是安全侧。
        """
        env = self.env
        dt = env.dt
        lim = np.radians(cfg.THETA_S_TARGET_DEG)
        hard = np.radians(cfg.THETA_S_LIMIT_DEG)
        self.goto(lim)                     # 严格归位到 +lim（不认"带内即成功"）
        accs, per = [], []
        for _ in range(cycles):
            for d in (-1.0, +1.0):
                # ---- 加速段：从 −d·lim 冲向中点，Ts = +d·τ_max ----
                t, rec, bad = 0.0, [], False
                while True:
                    _st, f = self._measure()
                    if abs(f[2]) > hard or t > 5.0:
                        bad = True
                        break
                    if d * f[2] >= 0.0:
                        break
                    rec.append((t, f[2]))
                    Tb, Ts = self._raw(d * self.tau_s_max, f)
                    env.step(Tb, Ts)
                    t += dt
                if not bad and len(rec) >= 8:
                    tt = np.array([r[0] for r in rec])
                    yy = np.array([r[1] for r in rec])
                    c2 = float(np.polyfit(tt, yy, 2)[0])
                    accs.append(2.0 * c2 * d)          # 乘 d 统一成正
                    per.append(2.0 * c2 * d)
                # ---- 制动段：从中点用 −d·τ_max 刹住 ----
                # 退出判据必须用**速度方向翻转**（d·θ̇ ≤ 0），不能卡 |θ̇|<0.03：
                # 减速度 ~40 rad/s² 时每帧 Δθ̇≈0.4 rad/s，会一步跨过那个窄带（实测跑飞）。
                t = 0.0
                while True:
                    _st, f = self._measure()
                    if d * f[3] <= 0.0 or d * f[2] >= lim or t > 5.0:
                        break
                    Tb, Ts = self._raw(-d * self.tau_s_max, f)
                    env.step(Tb, Ts)
                    t += dt
                # 刹停点通常到不了 ±lim（刹得比加速狠），下一趟前先严格归位，几何才一致
                if abs(self._measure()[1][2]) > hard:
                    break
                self.goto(d * lim)
        if len(accs) < 3:
            return dict(ok=False, n=len(accs), per=per)
        a_mean = float(np.mean(accs))
        self.a_max_cal = a_mean
        return dict(ok=True, n=len(accs), per=per, a_mean=a_mean,
                    a_std=float(np.std(accs)),
                    a_min=float(np.min(accs)), a_max=float(np.max(accs)))

    # ---------------- 等速往返：一个速度幅值下跑 SS_TIME_BUDGET 秒 ----------------
    def shuttle(self, omega_mag: float, amp: float):
        """在 ±amp 之间等速往返，返回 (逐趟记录列表, 测量半宽, 是否触碰限位)。

        参考速度**限斜坡**（SS_A_REF），不是从 +ω 一步跳到 −ω：
        反作用力矩 ≈ M12·θ̈_s，参考跳变会让小 yaw 用满 1 N·m 去换向，大 yaw
        被顶得乱晃（实测 sim 里 θ̇_b RMS 0.6 rad/s，窗口全废）。限斜坡后反作用
        力矩有界，抱死的大 yaw 跟得上。

        反向点：|θ_s| + ω²/(2·A_REF) ≥ amp（正好是斜坡所需距离），因此
        匀速段 = |θ_s| ≤ amp − ω²/(2·A_REF)，测量窗再取它中间的 SS_MEAS_FRAC。
        幅值上限就是 THETA_S_TARGET_DEG(±30°)。
        """
        env = self.env
        dt = env.dt
        a_ref = cfg.SS_A_REF
        hard = np.radians(cfg.THETA_S_LIMIT_DEG)
        ramp = omega_mag ** 2 / (2.0 * a_ref)
        cv_half = max(0.0, amp - ramp)
        gate = cfg.SS_MEAS_FRAC * cv_half
        seg, direction, segs, n_over, peak = 0, +1, [], 0, 0.0
        v_ref, target = 0.0, omega_mag
        armed = True          # 必须先回到带内、且确实朝当前方向运动，才允许下一次反向
        # 反向判据用的位置估计：一阶低通 + 相位补偿（滤噪声但不引入滞后）
        a_rev = cfg.SS_REV_LPF_ALPHA
        tau_lag = (1.0 - a_rev) / a_rev * dt
        th_f, dth_f = float(env.state().theta_s), float(env.state().dtheta_s)
        buf = dict(seg=[], sign=[], th_b=[], dth_b=[], th_s=[], dth_s=[], Ts=[], Tb=[],
                   v_ref=[])
        n_budget = int(round(cfg.SS_TIME_BUDGET / dt))
        # 当前趟是否越过 THETA_S_LIMIT_DEG：越过的**这一趟**整趟作废（电控在此介入
        # 削弱力矩，发出的力矩≠实际力矩），但**不丢弃这个速度**——其余趟照常采，
        # 速度继续跑（用户明确要求：超限只是一次性事件，丢那一条就行）。
        bad = False

        def _flush():
            nonlocal bad
            if len(buf["seg"]) and not bad:
                segs.append({k: np.asarray(v, dtype=np.float64) for k, v in buf.items()})
            for k in buf:
                buf[k] = []
            bad = False

        for _ in range(n_budget):
            st, f = self._measure()
            th_f += a_rev * (st.theta_s - th_f)
            dth_f += a_rev * (st.dtheta_s - dth_f)
            th = th_f + dth_f * tau_lag          # 相位补偿后的位置（用于反向判据）
            # ---- 参考速度斜坡（限加速度 ⇒ 反作用力矩有界）----
            v_prev = v_ref
            v_ref += float(np.clip(target - v_ref, -a_ref * dt, a_ref * dt))
            a_dem = (v_ref - v_prev) / dt
            # ---- 反向：实测位置 + 斜坡刹车距离（提前 ramp 触发，峰值才停在 amp；
            #      滞环：回带内且确实朝该方向运动才重新武装，抗位置噪声来回翻转）----
            if armed and direction > 0 and th + ramp >= amp:
                _flush()
                direction, seg, target, armed = -1, seg + 1, -omega_mag, False
            elif armed and direction < 0 and th - ramp <= -amp:
                _flush()
                direction, seg, target, armed = +1, seg + 1, omega_mag, False
            if not armed and abs(th) <= 0.4 * amp and direction * dth_f > 0.0:
                armed = True
            peak = max(peak, abs(st.theta_s))
            if abs(st.theta_s) > hard and not bad:
                bad = True                       # 只作废这一趟，不 break、不丢速度
                n_over += 1                      # 按**趟**计数（bad 只在翻转时加一次）
            Tb, Ts = self._torques(v_ref, f, a_dem)
            buf["seg"].append(seg); buf["sign"].append(direction)
            buf["th_b"].append(st.theta_b); buf["dth_b"].append(st.dtheta_b)
            buf["th_s"].append(st.theta_s); buf["dth_s"].append(st.dtheta_s)
            buf["Ts"].append(Ts); buf["Tb"].append(Tb); buf["v_ref"].append(v_ref)
            env.step(Tb, Ts)
        _flush()
        return segs, gate, n_over, peak


# ===========================================================================
# 逐趟验收 + 统计量
# ===========================================================================
def pass_row(seg: dict, gate: float, omega_mag: float, env, pair_id: int,
             pass_index: int, amp: float, hold_big: bool, tau_s_max: float):
    """把一段（一个方向的一趟）压缩成一行；返回 (行, 剔除原因)；合格时原因为 ""。

    验收（全部通过才留下）：
      * 只取 |θ_s| ≤ gate 的样本（中间匀速段；以 0 为中心对称 ⇒ 重力一阶抵消）。
        每趟都必须是从一个刹车点横穿到另一个的**完整趟**（θ_s 在窗内从 −gate
        穿到 +gate），否则半趟会把起步加速混进均值；
      * 稳态判据：前半均值 vs 后半均值 |v1−v2| ≤ 4·(σ_v/√n + frac·ω + abs)
        （同 gen_friction_sweep 的判据，窗口改成位置门控）；
      * 力矩未饱和；平均速度不能远低于目标；
      * θ̇_b / θ̈_b RMS 不超限（大 yaw 没固定住就丢）。
    """
    th_s, dth_s = seg["th_s"], seg["dth_s"]
    # 必须是**完整的一趟**：θ_s 在窗内从 −gate 穿到 +gate（或反向）。
    # 否则（会话第一趟往往从 θ_s=0 起步）窗只覆盖半侧，起步加速让该趟 ⟨θ̈_s⟩ 明显
    # 偏离 0，混进正/反均值里把 ΔTs 带偏（实测未加此判据时 ω=1 处偏差 ~24%）。
    if not (th_s.min() < -gate and th_s.max() > gate):
        return None, f"非完整趟（θ_s∈[{np.degrees(th_s.min()):.2f}, {np.degrees(th_s.max()):.2f}]° 未横穿 ±{np.degrees(gate):.2f}°）"
    sel = np.abs(th_s) <= gate
    n = int(sel.sum())
    if n < 5:
        return None, f"窗内样本 {n} < 5"
    v = dth_s[sel]
    h = n // 2
    tol = cfg.SS_VEL_TOL_FRAC * omega_mag + cfg.SS_VEL_TOL_ABS
    se = max(env.sigma[1], 1e-6) / np.sqrt(max(h, 1)) + tol
    if abs(v[:h].mean() - v[h:].mean()) > 4.0 * se:
        return None, (f"非稳态 |v1-v2|={abs(v[:h].mean()-v[h:].mean()):.3f} "
                      f"> 4·se={4*se:.3f}")
    Ts = seg["Ts"][sel]
    if np.abs(Ts).max() >= 0.999 * tau_s_max:
        return None, "力矩饱和"
    omega = float(v.mean())
    if abs(omega) < 0.2 * omega_mag:
        return None, f"速度太低 |ω|={abs(omega):.4f} < 0.2·{omega_mag:g}"
    # 窗口内波动：卡**均值标准误**（辨识用的是均值）；逐样本 std 只记录
    v_sem = float(v.std()) / np.sqrt(n)
    v_lim = max(cfg.SS_DTHETA_S_SEM_FRAC * abs(omega), cfg.SS_DTHETA_S_SEM_ABS)
    if v_sem > v_lim:
        return None, (f"速度均值标准误 {v_sem:.5f} > {v_lim:.5f} rad/s"
                      f"（λ·σ/√n={cfg.LAMBDA*v_sem:.2f}，n={n}）")
    t_sem = float(Ts.std()) / np.sqrt(n)
    t_lim = max(cfg.SS_TS_SEM_FRAC * abs(float(Ts.mean())), cfg.SS_TS_SEM_ABS)
    if t_sem > t_lim:
        return None, (f"力矩均值标准误 {t_sem:.5f} > {t_lim:.5f} N·m"
                      f"（Ts std={Ts.std():.4f}, n={n}）")
    dth_b = seg["dth_b"][sel]
    if float(np.sqrt(np.mean(dth_b ** 2))) > cfg.SS_DTHETA_B_RMS_MAX:
        return None, (f"θ̇_b RMS={np.sqrt(np.mean(dth_b**2)):.3f} "
                      f"> {cfg.SS_DTHETA_B_RMS_MAX:g}")
    dd_b = np.diff(dth_b) / env.dt if n > 1 else np.zeros(1)
    # 卡**平均**角加速度（进 Ts 的是 M12·⟨θ̈_b⟩，且会被辨识端扣掉）；
    # 逐样本 RMS 是带噪速度差分的噪声底，只在下面作为诊断记录。
    ddb_noise = float(np.sqrt(2.0) * max(env.sigma[1], 1e-9) / max(n - 1, 1) / env.dt)
    if abs(float(dd_b.mean())) > max(cfg.SS_DDTHETA_B_RATE_MAX, 5.0 * ddb_noise):
        return None, (f"大 yaw 平均角加速度 {abs(dd_b.mean()):.2f} "
                      f"> {max(cfg.SS_DDTHETA_B_RATE_MAX, 5.0*ddb_noise):.2f} rad/s²")

    th_b = seg["th_b"][sel]
    th = th_s[sel]
    psi_b, psi_s = th_b, th_b + th                     # 基座恒为 0
    dd = np.diff(dth_s[sel]) / env.dt if n > 1 else np.zeros(1)
    return dict(
        omega_ref=float(seg["sign"][0] * omega_mag), omega=omega,
        # 与大 yaw 脚本一致：对**窗口平均速度**取 tanh（速度测量有噪声时
        # mean(tanh(λv)) 会被 tanh 的非线性严重压缩，甚至均值趋于 0）
        tanh_omega=float(np.tanh(cfg.LAMBDA * omega)),
        tanh_omega_mean=float(np.tanh(cfg.LAMBDA * v).mean()),
        ts_mean=float(Ts.mean()), ts_std=float(Ts.std()),
        ts_sem=float(Ts.std()) / np.sqrt(n),
        ts_min=float(Ts.min()), ts_max=float(Ts.max()),
        tb_mean=float(seg["Tb"][sel].mean()),
        mean_sin_psi_b=float(np.sin(psi_b).mean()),
        mean_cos_psi_b=float(np.cos(psi_b).mean()),
        mean_sin_psi_s=float(np.sin(psi_s).mean()),
        mean_cos_psi_s=float(np.cos(psi_s).mean()),
        theta_s_mean_deg=float(np.degrees(th.mean())),
        theta_s_absmax_deg=float(np.degrees(np.abs(th).max())),
        theta_b_mean_deg=float(np.degrees(th_b.mean())),
        theta_b_std_deg=float(np.degrees(th_b.std())),
        dtheta_s_std=float(v.std()),
        dtheta_s_sem=float(v.std()) / np.sqrt(n),
        lam_dtheta_s_sem=float(cfg.LAMBDA * v.std() / np.sqrt(n)),
        dtheta_s_rms=float(np.sqrt(np.mean(v ** 2))),
        dtheta_b_rms=float(np.sqrt(np.mean(dth_b ** 2))),
        ddtheta_b_mean=float(dd_b.mean()),
        ddtheta_b_rms=float(np.sqrt(np.mean(dd_b ** 2))),
        ddtheta_s_mean=float(dd.mean()),
        ddtheta_s_rms=float(np.sqrt(np.mean(dd ** 2))),
        n_samples=float(n), pass_index=float(pass_index),
        pair_id=float(pair_id), direction=float(seg["sign"][0]),
        amp_deg=float(np.degrees(amp)), gate_deg=float(np.degrees(gate)),
        hold_big=np.float64(1.0 if hold_big else 0.0),
    ), ""


# ===========================================================================
# 速度 → 幅值规划
# ===========================================================================
def plan_amp(omega_mag: float) -> tuple[float, float, float]:
    """给定 |ω| → (往返幅值 A, 匀速对称半宽 cv_half, 测量窗半宽 gate) [rad]。

    A = clip(ω·SS_CV_TIME/2 + 斜坡距离, SS_AMP_MIN_DEG, THETA_S_TARGET_DEG−余量)
    —— 斜坡距离 ω²/(2·SS_A_REF) 必须算进去，否则高速时幅值全被换向斜坡吃掉。
    """
    a_cap = np.radians(cfg.THETA_S_TARGET_DEG)      # 主动控制范围就是 ±30°，不额外缩
    a_min = np.radians(cfg.SS_AMP_MIN_DEG)
    ramp = omega_mag ** 2 / (2.0 * cfg.SS_A_REF)
    amp = float(np.clip(omega_mag * cfg.SS_CV_TIME / 2.0 + ramp, a_min, a_cap))
    cv_half = max(0.0, amp - ramp)
    return amp, cv_half, cfg.SS_MEAS_FRAC * cv_half


# ===========================================================================
# 主流程
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None,
                    help="输出 npz 路径；时间戳会插进文件名（同 gen_friction_sweep）")
    ap.add_argument("--category", type=str, default=None,
                    help="类别名（= 目录名）：输出到 data/<类别>/sweep_s_<时间戳>.npz")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--zero-gravity", action="store_true",
                    help="[仿真] 把等效重力强制设为 0（水平面）；不给则随机")
    ap.add_argument("--gravity-seed", type=int, default=None)
    ap.add_argument("--no-noise", dest="noise", action="store_false",
                    help="[仿真] 关闭环境噪声")
    ap.add_argument("--sigma-pos", type=float, default=cfg.SIGMA_POS)
    ap.add_argument("--sigma-vel", type=float, default=cfg.SIGMA_VEL)
    ap.add_argument("--sigma-tau", type=float, default=cfg.SIGMA_TAU)

    # ---- 真实硬件环境（RealEnv）----
    ap.add_argument("--real", action="store_true",
                    help="用真实硬件环境（同 gen_friction_sweep 的 --real）")
    ap.add_argument("--imu-location", choices=("head", "big_yaw"), default="head")
    ap.add_argument("--spin-us", type=float, default=300.0)
    ap.add_argument("--tau-b-max", type=float, default=cfg.TAU_B_MAX)
    ap.add_argument("--tau-s-max", type=float, default=cfg.TAU_S_MAX)
    ap.add_argument("--gravity-settle-s", type=float, default=0.3)
    ap.add_argument("--ready-timeout-s", type=float, default=5.0)
    ap.add_argument("--hold-pitch", action="store_true")
    ap.add_argument("--pitch-target-deg", type=float, default=0.0)

    # ---- 本实验特有 ----
    ap.add_argument("--hold-big", action="store_true",
                    help="用位置环把大 yaw 抱在会话开始角度；不给则 Tb=0（人工/机械固定）")
    ap.add_argument("--no-gravity-ff", dest="gravity_ff", action="store_false",
                    default=cfg.SS_GRAVITY_FF,
                    help="关掉关节 s 的重力前馈 G2（做对照用；差分法本身不依赖它）")
    ap.add_argument("--ss-a-ref", type=float, default=cfg.SS_A_REF,
                    help="参考速度斜坡角加速度 [rad/s²]（越小反作用力矩越小、匀速段越短）")
    ap.add_argument("--ss-cv-time", type=float, default=cfg.SS_CV_TIME,
                    help="每个方向匀速段的目标时长 [s]（决定往返幅值）")
    ap.add_argument("--ss-time-budget", type=float, default=cfg.SS_TIME_BUDGET,
                    help="每个速度的总时间预算 [s]（决定往返趟数）")
    ap.add_argument("--ss-amp-min-deg", type=float, default=cfg.SS_AMP_MIN_DEG)
    ap.add_argument("--ss-min-samples", type=int, default=cfg.SS_MIN_SAMPLES)
    ap.add_argument("--ss-vel-tol-frac", type=float, default=cfg.SS_VEL_TOL_FRAC)
    ap.add_argument("--ss-vel-tol-abs", type=float, default=cfg.SS_VEL_TOL_ABS)
    ap.add_argument("--ss-dtheta-b-rms-max", type=float, default=cfg.SS_DTHETA_B_RMS_MAX)
    ap.add_argument("--ss-ddtheta-b-rate-max", type=float, default=cfg.SS_DDTHETA_B_RATE_MAX)
    ap.add_argument("--ss-dtheta-s-sem-frac", type=float, default=cfg.SS_DTHETA_S_SEM_FRAC)
    ap.add_argument("--ss-dtheta-s-sem-abs", type=float, default=cfg.SS_DTHETA_S_SEM_ABS)
    ap.add_argument("--ss-ts-sem-frac", type=float, default=cfg.SS_TS_SEM_FRAC)
    ap.add_argument("--ss-ts-sem-abs", type=float, default=cfg.SS_TS_SEM_ABS)
    ap.add_argument("--ss-rev-lpf-alpha", type=float, default=cfg.SS_REV_LPF_ALPHA)
    ap.add_argument("--goto-kp", type=float, default=cfg.SS_GOTO_KP,
                    help="归位位置 PID 的 Kp [N·m/rad]")
    ap.add_argument("--goto-ki", type=float, default=cfg.SS_GOTO_KI,
                    help="归位位置 PID 的 Ki [N·m/(rad·s)]")
    ap.add_argument("--goto-kd", type=float, default=cfg.SS_GOTO_KD,
                    help="归位位置 PID 的 Kd [N·m·s/rad]（作用在实测 θ̇_s 上）")
    ap.add_argument("--ss-repos-v-max", type=float, default=cfg.SS_REPOS_V_MAX,
                    help="归位参考轨迹限速 [rad/s]（越小起步越柔和、越慢）")
    ap.add_argument("--no-amax-cal", dest="amax_cal", action="store_false",
                    default=cfg.SS_AMAX_CAL,
                    help="跳过开始前用最大力矩标定可用角加速度这一步")
    ap.add_argument("--amax-cycles", type=int, default=cfg.SS_AMAX_CYCLES,
                    help="角加速度标定的往返次数（正反各一段/次，取平均）")
    ap.add_argument("--ss-a-ref-frac", type=float, default=cfg.SS_A_REF_FRAC,
                    help="换向斜坡取实测 a_max 的比例（默认 0.5，留一半力矩做余量）")
    ap.add_argument("--ss-kpv", type=float, default=cfg.SS_KPV,
                    help="θ_s 速度环比例增益")
    ap.add_argument("--ss-kiv", type=float, default=cfg.SS_KIV,
                    help="θ_s 速度环积分增益（低速能否破静摩擦主要看它）")
    ap.add_argument("--ss-lpf-alpha", type=float, default=cfg.SS_CTRL_LPF_ALPHA,
                    help="控制用状态的一阶低通系数（1=不滤波）")
    ap.add_argument("--debug", action="store_true", help="打印每趟被剔除的原因")
    args = ap.parse_args()

    # 内部统一读 cfg，命令行覆盖就直接改这里的模块级配置
    cfg.SS_A_REF = args.ss_a_ref
    cfg.SS_CV_TIME = args.ss_cv_time
    cfg.SS_TIME_BUDGET = args.ss_time_budget
    cfg.SS_AMP_MIN_DEG = args.ss_amp_min_deg
    cfg.SS_MIN_SAMPLES = args.ss_min_samples
    cfg.SS_VEL_TOL_FRAC = args.ss_vel_tol_frac
    cfg.SS_VEL_TOL_ABS = args.ss_vel_tol_abs
    cfg.SS_DTHETA_B_RMS_MAX = args.ss_dtheta_b_rms_max
    cfg.SS_DDTHETA_B_RATE_MAX = args.ss_ddtheta_b_rate_max
    cfg.SS_DTHETA_S_SEM_FRAC = args.ss_dtheta_s_sem_frac
    cfg.SS_DTHETA_S_SEM_ABS = args.ss_dtheta_s_sem_abs
    cfg.SS_TS_SEM_FRAC = args.ss_ts_sem_frac
    cfg.SS_TS_SEM_ABS = args.ss_ts_sem_abs
    cfg.SS_REV_LPF_ALPHA = args.ss_rev_lpf_alpha
    cfg.SS_GOTO_KP = args.goto_kp
    cfg.SS_GOTO_KI = args.goto_ki
    cfg.SS_GOTO_KD = args.goto_kd
    cfg.SS_REPOS_V_MAX = args.ss_repos_v_max
    cfg.SS_AMAX_CAL = args.amax_cal
    cfg.SS_AMAX_CYCLES = args.amax_cycles
    cfg.SS_A_REF_FRAC = args.ss_a_ref_frac
    cfg.SS_KPV = args.ss_kpv
    cfg.SS_KIV = args.ss_kiv
    cfg.SS_CTRL_LPF_ALPHA = args.ss_lpf_alpha
    tau_s_max = args.tau_s_max if args.real else cfg.TAU_S_MAX

    stamp = cfg.run_stamp()
    if args.out is not None:
        _o = Path(args.out)
        out_path = _o.parent / f"{_o.stem}_{stamp}{_o.suffix or '.npz'}"
    else:
        base = (cfg.category_dir(args.category) if args.category
                else (cfg.DATA_DIR_FRICTION_S_REAL if args.real
                      else cfg.DATA_DIR_FRICTION_S))
        out_path = base / f"sweep_s_{stamp}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    omegas = sorted({abs(float(w)) for w in cfg.SS_OMEGA_LIST if abs(float(w)) > 0.0})
    print(f"小 yaw 匀速往返摩擦采集: {'真实硬件' if args.real else '仿真'} 数据 -> {out_path}")
    print(f"  大 yaw: {'位置环抱死 θ_b' if args.hold_big else 'Tb=0（人工/机械固定）'}"
          f"   关节 s 重力前馈: {'开' if args.gravity_ff else '关'}")
    if cfg.SS_AMAX_CAL:
        print(f"  换向斜坡: 先用最大力矩标定可用角加速度，再取 min(--ss-a-ref="
              f"{cfg.SS_A_REF:g}, {cfg.SS_A_REF_FRAC:g}×实测)")
    if not args.hold_big:
        print("    ⚠ 全程大 yaw 力矩为 0，请先用机械/人工把大 yaw **完全固定**；"
              "θ̇_b/θ̈_b 超限的趟会被自动剔除")
    print(f"  速度表 [rad/s]: {omegas}")
    print(f"  幅值 A(ω)=clip(ω·{cfg.SS_CV_TIME:g}/2 + ω²/(2·a_ref), "
          f"{cfg.SS_AMP_MIN_DEG:g}°, "
          f"{cfg.THETA_S_TARGET_DEG:g}°)"
          f"   每速度 {cfg.SS_TIME_BUDGET:g}s  主动控制 {cfg.THETA_S_TARGET_DEG:g}°"
          f"（单趟越过 {cfg.THETA_S_LIMIT_DEG:g}° 只作废该趟）")

    if args.real:
        env = RealEnv(
            dt=cfg.DT,
            imu_location=(ImuLocation.ON_HEAD if args.imu_location == "head"
                          else ImuLocation.ON_BIG_YAW),
            tau_b_max=args.tau_b_max, tau_s_max=args.tau_s_max, spin_us=args.spin_us,
            gravity_settle_s=args.gravity_settle_s, hold_pitch=args.hold_pitch,
            pitch_target=float(np.radians(args.pitch_target_deg)),
            ready_timeout_s=args.ready_timeout_s)
    else:
        env = SimEnv(zero_gravity=args.zero_gravity, noise=args.noise,
                     sigma_pos=args.sigma_pos, sigma_vel=args.sigma_vel,
                     sigma_tau=args.sigma_tau, seed=args.gravity_seed)
    gx, gy = env.gravity
    print(f"  等效重力（{'反解得到' if args.real else '构造时随机确定'}）: "
          f"α={env.gravity_alpha_deg:.3f}°  (gx,gy)=({gx:.4f},{gy:.4f})  "
          f"|g|={np.hypot(gx, gy):.4f} m/s²")

    rows: list[dict] = []
    warned = []
    with env:
        loop = SweepLoop(env, hold_big=args.hold_big, gravity_ff=args.gravity_ff,
                         tau_s_max=tau_s_max)
        # ---- 开始记录前：用最大力矩标定可用角加速度，换向斜坡取它的 SS_A_REF_FRAC 倍 ----
        amax_meas = float("nan")
        if cfg.SS_AMAX_CAL:
            cal = loop.calibrate_amax(cfg.SS_AMAX_CYCLES)
            if cal["ok"]:
                amax_meas = cal["a_mean"]
                used = min(cfg.SS_A_REF, cfg.SS_A_REF_FRAC * amax_meas)
                print(f"  最大角加速度标定（{cal['n']} 段，τ_max={tau_s_max:g} N·m）: "
                      f"平均 {amax_meas:.1f} rad/s²（std {cal['a_std']:.2f}，"
                      f"范围 {cal['a_min']:.1f}~{cal['a_max']:.1f}）")
                print(f"    换向斜坡 = min(--ss-a-ref={cfg.SS_A_REF:g}, "
                      f"{cfg.SS_A_REF_FRAC:g}×{amax_meas:.1f}) = {used:.1f} rad/s²"
                      f"   （≈ τ/M22，反向峰值因此停在 ±{cfg.THETA_S_TARGET_DEG:g}° 内）")
                cfg.SS_A_REF = used
            else:
                safe = min(cfg.SS_A_REF, cfg.SS_A_REF_SAFE)
                print(f"  最大角加速度标定失败（只拿到 {cal['n']} 段，需 ≥3）！"
                      f"换向斜坡从 {cfg.SS_A_REF:g} 收到保守值 {safe:g} rad/s²"
                      f"（标定失败往往意味着力矩方向/标定有问题，先查那个）")
                cfg.SS_A_REF = safe
        for pair_id, w in enumerate(omegas):
            amp, _cv_half, gate = plan_amp(w)
            est = 2.0 * gate / (w * env.dt)          # 单趟预估样本数
            if est < cfg.SS_MIN_SAMPLES:
                print(f"  ω={w:+.4f}: 跳过——行程内匀速窗太小"
                      f"（A={np.degrees(amp):.2f}°, 预估 {est:.1f} 样本 < {cfg.SS_MIN_SAMPLES}）")
                continue
            # 严格归位到 −A（不接受"带内即成功"）：否则每换一个速度，第一趟会从
            # 上个速度留下的位置/速度出发，可能在幅值边界外继续冲，峰值白白多出
            # ω²/(2a)（实测 ω=3 首趟冲到 35.5°）。从静止的 −A 起步，几何才一致。
            if not loop.goto(-amp):
                print(f"  ω={w:+.4f}: 归位未达容差（仍在 {cfg.SS_REPOS_MAX_STEPS} 步内没到位），"
                      f"首趟可能被当作非完整趟丢掉")
            segs, gate, n_over, peak = loop.shuttle(w, amp)
            got = 0
            for i, seg in enumerate(segs):
                r, why = pass_row(seg, gate, w, env, pair_id, i, amp, args.hold_big,
                                  tau_s_max)
                if r is None:
                    if args.debug:
                        print(f"          趟{i}({'+' if seg['sign'][0] > 0 else '-'}) 剔除: {why}")
                    continue
                got += 1
                rows.append(r)
            tag = ""
            if got < cfg.SS_MIN_PASS:
                tag = f"  ← 不足 {cfg.SS_MIN_PASS} 趟，建议放宽容差/改 --hold-big/降速"
                warned.append(w)
            print(f"  ω={w:+.4f}: A={np.degrees(amp):.2f}°  窗±{np.degrees(gate):.2f}°  "
                  f"{got}/{len(segs)} 趟合格  |θ_s|峰 {np.degrees(peak):.1f}°{tag}")
            if n_over:
                print(f"          越过 {cfg.THETA_S_LIMIT_DEG:g}° 的 {n_over} 趟已作废"
                      f"（该速度其余趟照常保留，未中断）")
            elif np.degrees(peak) > cfg.THETA_S_TARGET_DEG + 1.0:
                print(f"          换向峰值 {np.degrees(peak):.1f}° 超过控制目标 "
                      f"{cfg.THETA_S_TARGET_DEG:g}°（离 {cfg.THETA_S_LIMIT_DEG:g}° 还有 "
                      f"{cfg.THETA_S_LIMIT_DEG - np.degrees(peak):.1f}°）：想压低就调小 "
                      f"--ss-a-ref({cfg.SS_A_REF:g}) 或 --ss-cv-time")
        if args.real:
            print(f"  帧统计: frames={env.frame_count} late={env.late_count} "
                  f"平均周期={env.mean_period*1e3:.3f}ms 最大={env.max_period*1e3:.3f}ms")

    if not rows:
        print("本次没有合格的往返趟")
        return 1

    keys = sorted(rows[0].keys())
    data = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}
    data.update(gx=float(gx), gy=float(gy), alpha_deg=float(env.gravity_alpha_deg),
                real=np.float64(1.0 if args.real else 0.0),
                joint=np.float64(1.0),                       # 1 = 关节 s（小 yaw）
                a_ref_used=float(cfg.SS_A_REF),              # 本次实际用的换向斜坡
                amax_meas=float(amax_meas),                  # 标定出的可用角加速度
                lambda_=float(cfg.LAMBDA))
    np.savez_compressed(out_path, **data)

    # ---- 现场自检：模型无关的正反差分最小二乘（不需要任何动力学参数）----
    print(f"\n完成：{len(rows)} 趟合格（{len({int(r['pair_id']) for r in rows})} 个速度）"
          f" -> {out_path}")
    if warned:
        print(f"  注意：这些速度的合格趟数不足（{', '.join(f'{w:g}' for w in warned)}），"
              f"差分对可能偏少")
    check = quick_diff_fit(rows)
    if check is not None:
        fsc, fsv, npair, resid = check
        print(f"  现场自检（正反差分，不需要已辨识参数）: fsc={fsc:.5f}  fsv={fsv:.5f}"
              f"   用了 {npair} 对，残差 {resid:.2e} N·m")
    print(f"  |θ_s| 峰值 {data['theta_s_absmax_deg'].max():.3f}°  "
          f"θ̇_b RMS 中位 {np.median(data['dtheta_b_rms']):.2e} rad/s  "
          f"θ̈_b RMS 中位 {np.median(data['ddtheta_b_rms']):.2e} rad/s²")
    return 0


def quick_diff_fit(rows: list[dict]):
    """把正反趟配对做一次二维最小二乘，仅用于现场判断数据好坏（模型无关）。"""
    pairs: dict[int, dict[int, list[dict]]] = {}
    for r in rows:
        pairs.setdefault(int(r["pair_id"]), {}).setdefault(int(r["direction"]), []).append(r)
    A, y = [], []
    for pid in sorted(pairs):
        d = pairs[pid]
        if 1 not in d or -1 not in d:
            continue
        pp, pm = d[1], d[-1]
        db = (np.mean([r["theta_b_mean_deg"] for r in pp])
              - np.mean([r["theta_b_mean_deg"] for r in pm]))
        if abs(db) > np.degrees(cfg.SS_DDB_MEAN_MAX):
            continue
        mp = lambda rs, k: float(np.mean([r[k] for r in rs]))       # noqa: E731
        A.append([mp(pp, "omega") - mp(pm, "omega"),
                  mp(pp, "tanh_omega") - mp(pm, "tanh_omega")])
        y.append(mp(pp, "ts_mean") - mp(pm, "ts_mean"))
    if len(A) < 3:
        return None
    A = np.asarray(A)
    y = np.asarray(y)
    (fsv, fsc), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(fsc), float(fsv), int(len(y)), float(np.std(y - A @ np.array([fsv, fsc])))


if __name__ == "__main__":
    raise SystemExit(main())

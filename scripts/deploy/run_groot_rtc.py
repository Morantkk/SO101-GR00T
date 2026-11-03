#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RTC 版推理客户端（流式重叠 + 前缀约束）—— 与基线 `run_groot_official.py` **并存**。

基线脚本一个字不改，本文件直接 `import run_groot_official as B` 复用它的全部工具
（建机器人 / 取状态 / JPEG 编码 / 限幅 / 录制 / 复位位姿），只换掉执行环。

基线的执行环是**同步**的：
    请求（阻塞 0.3~0.5s，臂不动）→ 拿整块 16 步 → 顺序执行完 → 再请求
    ⇒ 每个块边界都有一次停顿，且新块的起点与旧块的终点在时间上断开。

本脚本改成**流式**：
    开始执行新块的第 1 步时，就异步发出"下一块"的请求（带 `__rtc.d = 已执行步数`）；
    请求回来时（通常在第 6~8 步），从**当前位置**切到新块继续跑。
    服务端（`101_policy_server_rtc.py`）会把新块的前 d 步钉死在我们已经执行过的轨迹上，
    所以切换点上不会跳变。

用法（先起服务端，再拉隧道，然后本机跑）:
    <py> scripts/run_groot_rtc.py --steps 400 --task "Grab the pen and put it into the box"
    <py> scripts/run_groot_rtc.py --steps 400 --no-rtc      # 同一份代码跑对照组

★ 与基线对比时要看的是：块切换点前后的关节指令差（日志里 `[切换]` 那行）、
  实际动作速率（结尾会打）、以及录下来的视频里有没有那一下顿挫。
"""
import argparse
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_groot_official as B  # noqa: E402  ★ 基线当库用，不改它


def smooth_chunk(arr, w):
    """块内**对称**滑动平均（同时参考未来几步 ⇒ 不产生滞后；EMA 才有滞后）。

    专治模型逐帧 1~2° 的高频噪声：窗口 5 噪声约降到 1/sqrt(5)。
    """
    if not w or w < 3 or len(arr) < w:
        return arr
    k = w // 2
    pad = np.pad(np.asarray(arr, dtype=np.float64), ((k, k), (0, 0)), mode="edge")
    ker = np.ones(w) / float(w)
    return np.stack([np.convolve(pad[:, j], ker, "valid") for j in range(pad.shape[1])], axis=1)


def build_obs(B_, obs, task, jpeg_quality, d=None, reset=False):
    """与基线 `main()` 里完全同构的 obs_dict，额外挂 RTC 控制键。"""
    st = B_.state_vec(obs)
    od = {
        "video.front": B_.to_wire(B_.prep_image(obs["front"], "front"), jpeg_quality),
        "video.wrist": B_.to_wire(B_.prep_image(obs["wrist"], "wrist"), jpeg_quality),
        "state.single_arm": st[:5][None, :],
        "state.gripper": st[5:6][None, :],
        "annotation.human.task_description": [task],
    }
    if reset:
        od["__rtc.reset"] = True
    elif d is not None and d > 0:
        od["__rtc.d"] = int(d)
    return st, od


class Fetcher:
    """一次在飞的异步推理请求。

    ★ 每次都用**自己的一条 ZMQ 连接**。原因：REQ socket 既不线程安全，
      也不允许"发出去还没收到就再发"——那样会抛
      `ZMQError: Operation cannot be accomplished in current state`（EFSM）。
      2026-10-30 实测：主线程和后台线程共用一个 socket，在"块用尽"那一刻必然崩。
    """

    def __init__(self, host, port, obs_dict, d_at=0, timeout_ms=30000):
        self.box = {}
        self.d_at = d_at                      # 这次请求带给服务端的 d（已执行步数）
        self.cli = B.ZmqPolicyClient(host, port, timeout_ms=timeout_ms)
        self.obs = obs_dict
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        t0 = time.time()
        try:
            self.box["res"], self.box["nb"] = self.cli.get_action(self.obs)
        except Exception as e:  # noqa: BLE001
            self.box["err"] = e
        finally:
            self.box["lat"] = time.time() - t0
            try:
                self.cli.sock.close()
                self.cli.ctx.term()
            except Exception:  # noqa: BLE001
                pass

    def done(self):
        return not self.th.is_alive()

    def wait(self):
        """阻塞等到回包（块用尽时用它接上，避免同一个 socket 连发两次请求）。"""
        self.th.join()
        return self.take()

    def take(self):
        if "err" in self.box:
            raise self.box["err"]
        return self.box.get("res"), self.box.get("nb", 0), self.box.get("lat", 0.0)


def parse_chunk(res, clamp=True):
    sa = np.asarray(res["action.single_arm"], dtype=np.float64)
    sg = np.asarray(res["action.gripper"], dtype=np.float64)
    c = np.concatenate([sa.reshape(len(sa), -1), sg.reshape(len(sg), -1)], axis=1)
    hit = np.zeros(c.shape[1], dtype=int)
    if clamp and B.CLAMP_MARGIN is not None:
        c, hit = B.clamp_chunk(c)
    return c, hit


def run_stream_rtc(robot, cli, a):
    """流式执行环。返回 0/1。"""
    import cv2  # noqa: F401  仅为与基线保持可选依赖一致

    srv_host, srv_port = a.server.split(":")
    srv_port = int(srv_port)

    REC = B._new_recorder(a) if a.record_video else None
    csv_f = None
    if a.log_csv:
        d = os.path.dirname(os.path.abspath(a.log_csv))
        if d:
            os.makedirs(d, exist_ok=True)
        csv_f = open(a.log_csv, "w", encoding="utf-8")
        csv_f.write("step,chunk_idx,chunk_id,lat_s,kb,"
                    + ",".join("fb_" + m for m in B.MOTORS) + ","
                    + ",".join("cmd_" + m for m in B.MOTORS) + "\n")

    done = 0
    n_infer = 0
    lat_sum = 0.0
    payload_sum = 0
    n_switch = 0
    switch_gap = []
    clamp_total = np.zeros(len(B.MOTORS), dtype=int)
    t_start = time.time()

    cur = None          # 当前执行的块 [C,6]
    ci = 0              # 当前块已执行步数
    chunk_id = 0
    inflight = None
    first = True
    last_cmd = None      # 低通滤波的上一帧指令

    def fetch_blocking(obs, d=None, reset=False):
        st, od = build_obs(B, obs, a.task, a.jpeg_quality, d=d, reset=reset)
        t0 = time.time()
        res, nb = cli.get_action(od)
        return st, res, nb, time.time() - t0

    try:
        while a.steps <= 0 or done < a.steps:
            if cur is None:
                obs = robot.get_observation()
                st, res, nb, lat = fetch_blocking(obs, d=None, reset=first)
                first = False
                n_infer += 1
                lat_sum += lat
                payload_sum += nb
                cur, hit = parse_chunk(res, a.joint_clamp)
                clamp_total += hit
                cur = smooth_chunk(cur, a.chunk_smooth)
                ci = 0
                chunk_id += 1
                print("[取块#%d] 延迟 %.3fs 上行 %.0fKB chunk=%s"
                      % (chunk_id, lat, nb / 1024.0, cur.shape))

            # ★ 不再每步读观测（两份相机 ~13ms/步，纯浪费）：只在要发请求时才读。
            #   动作块本来就是开环执行的，中间不读也不影响行为，但控制频率能回到 ~14Hz。

            # 执行当前块的一步
            cmd = cur[ci]
            # ★ 一阶低通（EMA）：模型每步的输出有 1~2° 的高频噪声（实测指令反向率 40~58%），
            #   直接下发会看见"细抖"。α 越小越平滑、但滞后越大（α=0.6 约滞后 0.7 步）。
            if a.smooth and a.smooth < 1.0:
                if last_cmd is None:
                    last_cmd = np.array(cmd, dtype=np.float64)
                else:
                    last_cmd = a.smooth * np.asarray(cmd, dtype=np.float64) + (1.0 - a.smooth) * last_cmd
                cmd = last_cmd
            if not a.dry_run:
                robot.send_action(B.action_to_dict(cmd))
            if REC is not None:
                REC.snap(robot)
            if csv_f:
                fb = B.read_positions(robot)
                csv_f.write("%d,%d,%d,%.4f,%.1f," % (done, ci, chunk_id, lat, payload_sum / 1024.0)
                            + ",".join("%.3f" % v for v in (fb if fb is not None else st)) + ","
                            + ",".join("%.3f" % v for v in cmd) + "\n")
                csv_f.flush()
            time.sleep(a.dt)
            done += 1
            ci += 1

            # 到发射点 → 异步请求下一块（把"已执行步数"报给服务端）
            if a.rtc and inflight is None and ci >= a.rtc_fire and (a.steps <= 0 or done < a.steps):
                obs = robot.get_observation()          # 只在请求时读一次观测
                st = B.state_vec(obs)
                _, od = build_obs(B, obs, a.task, a.jpeg_quality, d=ci)
                inflight = Fetcher(srv_host, srv_port, od, d_at=ci)

            # 回包 → 从当前位置切块
            if inflight is not None and inflight.done():
                res, nb, lat = inflight.take()
                d_at = inflight.d_at
                inflight = None
                n_infer += 1
                lat_sum += lat
                payload_sum += nb
                new, hit = parse_chunk(res, a.joint_clamp)
                clamp_total += hit
                new = smooth_chunk(new, a.chunk_smooth)
                # ★ 对齐约定：新块 index i ↔ 时间 (d_at + i)（服务端按 d 平移前缀）
                #   现在到了第 ci 步 ⇒ 从新块第 (ci - d_at) 步继续。
                skip = max(0, min(ci - d_at, len(new) - 1))
                gap = float(np.max(np.abs(new[skip] - cur[min(ci, len(cur) - 1)])))
                if a.max_seam and a.max_seam > 0 and gap > a.max_seam:
                    # ★ 兜底：软引导失效（模型这一块跑偏了）时别硬切，否则臂会突然甩一下。
                    #   放弃这块，继续走旧计划；下一步会重新请求。
                    inflight = None
                    print("[拒切] 接缝差 %.1f° > 阈值 %.0f°，继续用旧计划" % (gap, a.max_seam))
                else:
                    n_switch += 1
                    switch_gap.append(gap)
                    print("[切换] 执行 %d 步后换块（延迟 %.3fs, d=%d）→ 从新块第 %d 步继续，"
                          "接缝处指令差 %.2f°" % (ci, lat, d_at, skip, gap))
                    cur = new
                    ci = skip
                    chunk_id += 1

            # 块用尽 → 必须从**当前**状态重新要一块（新鲜，不带 d）
            #   ★ 不能拿在飞的那块来续：它是 d 步之前的观测算出来的，等它回来时已经过期，
            #     拿它续会出现"每块只取 1 步、无限补块"的退化（实测 48 步里补了 33 次）。
            #   ★ 也不能在同一个 REQ socket 上再发一次（EFSM 崩）。
            #     所以：丢弃在飞的那块（它自己会收尾并关掉自己的 socket），用主连接阻塞取新块。
            if ci >= len(cur) and (a.steps <= 0 or done < a.steps):
                if inflight is not None:
                    inflight = None          # 作废；后台线程自行结束
                obs = robot.get_observation()
                st, res, nb, lat = fetch_blocking(obs, d=None, reset=False)
                n_infer += 1
                lat_sum += lat
                payload_sum += nb
                cur, hit = parse_chunk(res, a.joint_clamp)
                clamp_total += hit
                cur = smooth_chunk(cur, a.chunk_smooth)
                ci = 0
                chunk_id += 1
                print("[续块#%d] 延迟 %.3fs 上行 %.0fKB（块用尽，重新取块）"
                      % (chunk_id, lat, nb / 1024.0))
    except KeyboardInterrupt:
        print("\n(手动中断)")
    finally:
        if csv_f:
            csv_f.close()
        if REC is not None:
            REC.close()
        try:
            robot.disconnect()
        except Exception:
            pass
        dt = time.time() - t_start
        print("\n结束：执行 %d 个动作、推理 %d 次、切块 %d 次，用时 %.1fs"
              % (done, n_infer, n_switch, dt))
        if n_infer:
            print("      平均推理延迟 %.3fs、平均上行 %.0fKB"
                  % (lat_sum / n_infer, payload_sum / n_infer / 1024.0))
        if not a.dry_run and done:
            print("      实际动作速率 %.2f 动作/秒（目标 %d）" % (done / max(dt, 1e-9), a.cam_fps))
        if switch_gap:
            print("      切块接缝指令差：中位 %.2f°  最大 %.2f°（RTC 生效时应该很小）"
                  % (float(np.median(switch_gap)), float(np.max(switch_gap))))
        if clamp_total.any():
            print("      行程限幅生效：%s"
                  % ", ".join("%s %d" % (B.MOTORS[j][:11], int(clamp_total[j]))
                              for j in range(len(B.MOTORS)) if clamp_total[j]))
    return 0


def port_busy_hint(port):
    """串口打不开时，指出大概是哪个进程占着（Windows 上最常见：上次的客户端没退干净）。"""
    print("\n!! 串口 %s 打不开：PermissionError(13, 拒绝访问) = 端口存在但**已被别的进程占用**。" % port)
    print("   多半是上一次的 run_groot_rtc.py / run_groot_official.py 没退干净（杀外壳不会杀 python 子进程）。")
    try:
        import psutil

        hits = []
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            nm = (p.info["name"] or "").lower()
            if "python" not in nm:
                continue
            cl = " ".join(p.info["cmdline"] or [])
            if "groot" in cl or "lerobot" in cl or "ssh_tunnel" in cl:
                hits.append((p.info["pid"], cl[:100]))
        if hits:
            print("   怀疑这些进程（确认后按 PID 杀）：")
            for pid, cl in hits:
                print("     %6d  %s" % (pid, cl))
            print("   一条命令全清： taskkill /F /PID " + " /PID ".join(str(p) for p, _ in hits))
            print("")
            return
    except ImportError:
        pass

    # 没有 psutil 时的兜底：tasklist 只给 PID/内存，能看出有几个 python 在挂着
    import subprocess

    try:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"],
                           capture_output=True, timeout=10)
        # 中文 Windows 的 tasklist 输出是 GBK，不能按 utf-8 解
        out = r.stdout.decode("gbk", "replace")
        lines = [ln.strip() for ln in out.splitlines() if ln.strip().lower().startswith("python")]
    except Exception:  # noqa: BLE001
        lines = []
    if lines:
        print("   当前挂着的 python 进程（无法区分是谁，按内存占用猜一般是大那个）：")
        for ln in lines:
            print("     " + ln)
        print("   清理： taskkill /F /PID <pid>")
    else:
        print("   没扫到 python 进程；用管理员权限看谁占着：handle.exe %s（Sysinternals）" % port)
    print("")


def main():
    ap = argparse.ArgumentParser(description="GR00T + RTC 流式推理客户端")
    ap.add_argument("--robot-port", default="COM3")
    ap.add_argument("--robot-id", default="my_awesome_follower_arm")
    ap.add_argument("--front-idx", type=int, default=1)
    ap.add_argument("--wrist-idx", type=int, default=0)
    ap.add_argument("--front-size", default="640x360")
    ap.add_argument("--wrist-size", default="1280x720")
    ap.add_argument("--cam-fps", type=int, default=15)
    ap.add_argument("--server", default="127.0.0.1:5555")
    ap.add_argument("--task", default="Put all the objects on the table into the box")
    ap.add_argument("--steps", type=int, default=400, help="0 或负数 = 不限时间，跑到 Ctrl-C")
    ap.add_argument("--max-rel", type=float, default=8.0,
                    help="★ 每步最大关节变化（度），交给 lerobot 裁剪；0=不限速（会甩动）")
    ap.add_argument("--chunk-smooth", type=int, default=5,
                    help="★ 块内对称滑动平均窗口（奇数，1=关）。无滞后，专治逐帧高频噪声")
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="★ 指令一阶低通 α：1.0=不平滑，0.6=温和（默认），0.4=更平滑但更滞后")
    ap.add_argument("--max-seam", type=float, default=25.0,
                    help="★ 换块时若接缝差超过这个度数就不切（继续走旧计划，下一步再请求）")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--rtc", dest="rtc", action="store_true", default=True)
    ap.add_argument("--no-rtc", dest="rtc", action="store_false", help="对照组：不发 __rtc.* 键")
    ap.add_argument("--rtc-fire", type=int, default=1, help="执行到第几步就异步请求下一块")
    ap.add_argument("--joint-clamp", dest="joint_clamp", action="store_true", default=True)
    ap.add_argument("--no-joint-clamp", dest="joint_clamp", action="store_false")
    ap.add_argument("--record-video", default="", help="录 mp4 的路径（留空=不录）")
    ap.add_argument("--record-keys", default="front", help="默认只录第三视角；要看腕部写 front,wrist")
    ap.add_argument("--record-raw", action="store_true")
    ap.add_argument("--record-fps", type=float, default=0.0)
    ap.add_argument("--record-fourcc", default="mp4v")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--log-csv", default="")
    ap.add_argument("--dry-run", action="store_true", help="不真发动作，只看块与接缝")
    a = ap.parse_args()

    a.front_size = B.parse_size(a.front_size)
    a.wrist_size = B.parse_size(a.wrist_size)
    a.dt = 1.0 / float(a.cam_fps)

    host, port = a.server.split(":")
    cli = B.ZmqPolicyClient(host, int(port))
    print("ping ->", cli.ping())

    if a.dry_run:
        print("⚠ dry-run：不下发动作，但请求/切块照走 —— 只验状态机与协议，接缝差无物理意义")

    robot = B.build_robot(a)
    try:
        robot.connect(calibrate=not a.no_calibrate)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "拒绝访问" in msg or "PermissionError" in msg or "Access is denied" in msg:
            port_busy_hint(a.robot_port)
        print("!! 连不上机器人：%r" % (e,))
        print("   dry-run 可以加 --dry-run 跳过真机（仍会尝试构造 robot 对象）")
        return 2

    # ★ 每步限速：lerobot 会在 send_action 时把指令裁到「当前位置 ± max_rel 度」。
    #   基线脚本一直有这个闸门，RTC 版之前漏了 ⇒ 换块时 20~40° 的差会被一次性执行出去，
    #   表现就是机械臂"突然甩一下"。这里补上，默认 8°/步（≈120°/s @15Hz）。
    if a.max_rel and a.max_rel > 0:
        robot.config.max_relative_target = a.max_rel
        print("已设置 max_relative_target = %.1f 度/步（≈ %.0f°/s @15Hz）"
              % (a.max_rel, a.max_rel * a.cam_fps))
    else:
        print("⚠ 未设 max_relative_target（不限速，换块时可能甩动）")
    try:
        input(">>> 清场确认，按 Enter 开始（Ctrl-C 随时中断）…")
    except EOFError:
        pass
    return run_stream_rtc(robot, cli, a)


if __name__ == "__main__":
    sys.exit(main())

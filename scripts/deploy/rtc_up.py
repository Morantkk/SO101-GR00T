#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""确保「本机端口 → 云端 5555」的隧道是**真的能用**，不能用就自动重建。

为什么不能只看端口：paramiko 隧道的 SSH 通道可能已经死了，但本地监听端口还在，
新的 TCP 连接照样能建立，只是数据转发不出去 ⇒ 客户端表现为 zmq `Again`（连上但没回复）。
所以判据必须是「真打一次 ping」，不是 netstat。

用法:
    python rtc_up.py                    # 检查 127.0.0.1:5555，坏了就重建
    python rtc_up.py --local 5556       # 换本地端口
    python rtc_up.py --force            # 无条件重建
退出码: 0 = 通（可能是重建后），1 = 重建失败
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def ping(local_port, timeout_ms=2500):
    """返回 (ok, 详情)。"""
    try:
        import msgpack
        import zmq

        ctx = zmq.Context()
        s = ctx.socket(zmq.REQ)
        s.setsockopt(zmq.RCVTIMEO, timeout_ms)
        s.setsockopt(zmq.SNDTIMEO, timeout_ms)
        s.setsockopt(zmq.LINGER, 0)
        s.connect("tcp://127.0.0.1:%d" % local_port)
        s.send(msgpack.packb({"endpoint": "ping"}))
        try:
            r = msgpack.unpackb(s.recv())
            return True, str(r)
        finally:
            s.close()
            ctx.term()
    except Exception as e:  # noqa: BLE001
        return False, "%s: %s" % (type(e).__name__, e)


def kill_stale_tunnels():
    """杀掉所有命令行里带 ssh_tunnel 的 python 进程（就是本脚本起的那种）。"""
    try:
        import psutil
    except ImportError:
        print("  (无 psutil，跳过清理)")
        return 0
    n = 0
    me = os.getpid()
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.info["pid"] == me:
            continue
        if "python" not in (p.info["name"] or "").lower():
            continue
        cl = " ".join(p.info["cmdline"] or [])
        if "ssh_tunnel" in cl:
            try:
                p.kill()
                p.wait(timeout=5)
                n += 1
                print("  killed stale tunnel pid=%d" % p.info["pid"])
            except Exception as e:  # noqa: BLE001
                print("  kill pid=%d failed: %r" % (p.info["pid"], e))
    return n


def start_tunnel(local_port, remote, ssh_port, password, host):
    env = dict(os.environ)
    env["GR00T_PORT"] = str(ssh_port)
    env["GR00T_PASSWORD"] = password
    env["GR00T_HOST"] = host
    log = open(os.path.join(ROOT, "outputs", "tunnel.log"), "ab") if os.path.isdir(
        os.path.join(ROOT, "outputs")) else subprocess.DEVNULL
    flags = 0
    if os.name == "nt":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP：关掉当前窗口也不影响隧道
        flags = 0x00000008 | 0x00000200
    p = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "ssh_tunnel.py"),
         "--local", str(local_port), "--remote", remote],
        cwd=ROOT, env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
        creationflags=flags,
    )
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", type=int, default=5555)
    ap.add_argument("--remote", default="127.0.0.1:5555")
    ap.add_argument("--host", default=os.environ.get("GR00T_HOST", "connect.bjb1.seetacloud.com"))
    ap.add_argument("--ssh-port", default=os.environ.get("GR00T_PORT", "22"))
    ap.add_argument("--password", default=os.environ.get("GR00T_PASSWORD", ""))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--wait", type=float, default=30.0)
    a = ap.parse_args()

    ok, info = ping(a.local)
    print("ping 127.0.0.1:%d -> %s   %s" % (a.local, "OK" if ok else "FAIL", info))
    if ok and not a.force:
        return 0

    if not a.password:
        print("!! 没有密码：请先 set GR00T_PASSWORD=... 或传 --password")
        return 1

    print("重建隧道 ...")
    kill_stale_tunnels()
    time.sleep(1.0)
    start_tunnel(a.local, a.remote, a.ssh_port, a.password, a.host)

    t0 = time.time()
    while time.time() - t0 < a.wait:
        time.sleep(1.5)
        ok, info = ping(a.local, timeout_ms=2000)
        if ok:
            print("隧道已恢复（%.0f s）  ping -> %s" % (time.time() - t0, info))
            return 0
    print("!! 隧道重建后仍然 ping 不通（%.0f s）。可能：SSH 密码/端口变了，或云端服务端没起。"
          % a.wait)
    print("   看日志： outputs\\tunnel.log")
    return 1


if __name__ == "__main__":
    sys.exit(main())

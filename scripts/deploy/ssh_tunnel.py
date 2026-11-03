# -*- coding: utf-8 -*-
"""本地端口转发（paramiko 版 `ssh -L`），供「云端 policy server + 本机 robot client」使用。

背景：AutoDL 只把 SSH 端口映射到公网，8080 之类的自定义端口从外面连不上。
所以客户端要连的地址写成 127.0.0.1:<local_port>，由本脚本转发到云端的 <remote_host>:<remote_port>。

用法：
    # 把本机 8080 转发到 云端 127.0.0.1:8080
    C:/Users/Fakun/.workbuddy/binaries/python/envs/default/Scripts/python.exe ssh_tunnel.py

    # 自定义
    ... ssh_tunnel.py --local 8080 --remote 127.0.0.1:8080

保持前台运行；Ctrl-C 退出。（连接信息来自同目录 conn.py）
"""

import argparse
import os
import select
import socketserver
import sys
import threading
import time

import paramiko

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conn import HOST, PASSWORD, PORT as SSH_PORT, USER


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        transport = self.server.transport
        try:
            chan = transport.open_channel(
                "direct-tcpip",
                (self.server.chain_host, self.server.chain_port),
                self.request.getpeername(),
            )
        except Exception as e:  # noqa: BLE001
            print("[tunnel] open_channel failed: %s" % e, file=sys.stderr)
            return
        if chan is None:
            print("[tunnel] no channel", file=sys.stderr)
            return
        try:
            while True:
                r, _, _ = select.select([self.request, chan], [], [])
                if self.request in r:
                    data = self.request.recv(65536)
                    if len(data) == 0:
                        break
                    chan.sendall(data)
                if chan in r:
                    data = chan.recv(65536)
                    if len(data) == 0:
                        break
                    self.request.sendall(data)
        except Exception:
            pass
        finally:
            try:
                chan.close()
            except Exception:
                pass
            try:
                self.request.close()
            except Exception:
                pass


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", type=int, default=8080, help="本机监听端口")
    ap.add_argument("--bind", default="127.0.0.1", help="本机绑定地址")
    ap.add_argument("--remote", default="127.0.0.1:8080", help="云端目标 host:port")
    args = ap.parse_args()

    r_host, r_port = args.remote.rsplit(":", 1)
    r_port = int(r_port)

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        HOST,
        port=SSH_PORT,
        username=USER,
        password=PASSWORD,
        timeout=30,
        allow_agent=False,
        look_for_keys=False,
        banner_timeout=60,
        auth_timeout=60,
    )
    tr = cli.get_transport()
    tr.set_keepalive(30)

    srv = _Server((args.bind, args.local), _Handler)
    # 注意：Handler 里用 self.server.* 取值，所以必须挂在 **server 实例** 上（挂类上会导致
    # handle() 抛 AttributeError、连接被立刻关闭，客户端表现为 gRPC "End of TCP stream"）
    srv.chain_host = r_host
    srv.chain_port = r_port
    srv.transport = tr
    print(
        "[tunnel] %s:%d  ->  %s@%s:%d  ->  %s:%d   (Ctrl-C 退出)"
        % (args.bind, args.local, USER, HOST, SSH_PORT, r_host, r_port)
    )
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        while True:
            time.sleep(1)
            if not tr.is_active():
                print("[tunnel] SSH 连接已断开", file=sys.stderr)
                break
    except KeyboardInterrupt:
        print("\n[tunnel] bye")
    finally:
        srv.shutdown()
        cli.close()


if __name__ == "__main__":
    main()

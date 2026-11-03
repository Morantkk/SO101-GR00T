"""远程执行单条命令（Windows 无 sshpass 时的替代方案）。

用法:
    <venv-python> remote_exec.py "<远程命令>" [timeout]

连接信息统一放在同目录 conn.py（换实例只改那一个文件；import 不到时用下面的兜底默认值）。
"""
import os
import sys
import time

import paramiko

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from conn import HOST, PASSWORD, PORT, USER
except ImportError:  # 兜底：conn.py 不存在时
    HOST = "connect.bjb1.seetacloud.com"
    PORT = 43912
    USER = "root"
    PASSWORD = "FKNErOycPYzN"


def connect_with_retry(retries=5, delay=8):
    last = None
    for i in range(retries):
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                        timeout=30, banner_timeout=60, auth_timeout=60,
                        allow_agent=False, look_for_keys=False)
            return cli
        except Exception as e:  # noqa: BLE001
            last = e
            sys.stderr.write("[retry %d/%d] connect failed: %r\n" % (i + 1, retries, e))
            time.sleep(delay)
    raise last


def main():
    if len(sys.argv) < 2:
        print("usage: remote_exec.py '<cmd>' [timeout]")
        return 2
    cmd = sys.argv[1]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 300

    cli = connect_with_retry()
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout, get_pty=True)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    sys.stdout.write(out)
    if err.strip():
        sys.stdout.write("\n[STDERR]\n" + err)
    sys.stdout.write("\n[EXIT CODE] %s\n" % rc)
    cli.close()
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

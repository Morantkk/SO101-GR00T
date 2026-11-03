"""远端 GPU 机器的连接配置（示例）。

用法：把本文件复制成 ``conn.py`` 并填上你自己的信息，或者完全不动它、
直接用环境变量覆盖（所有脚本都优先读环境变量）：

    # PowerShell
    $env:GR00T_HOST="connect.xxx.seetacloud.com"
    $env:GR00T_PORT="40009"
    $env:GR00T_PASSWORD="your-password"
    python scripts/tools/remote_exec.py "nvidia-smi"

    # cmd
    set GR00T_HOST=connect.xxx.seetacloud.com
    set GR00T_PORT=40009
    set GR00T_PASSWORD=your-password
    python scripts\tools\remote_exec.py "nvidia-smi"

ssh_tunnel.py / remote_exec.py / upload_file.py / parallel_upload_checked.py /
parallel_download.py / dashboard_gr00t.py 都会 ``from conn import HOST, PORT, ...``，
import 不到时回落到各自文件里的默认值（同样可用环境变量覆盖）。
"""

import os

# ↓↓↓ 换成你自己的实例信息（或者干脆不填，全部用环境变量） ↓↓↓
_DEFAULT_HOST = os.environ.get("GR00T_HOST", "connect.example.seetacloud.com")
_DEFAULT_PORT = int(os.environ.get("GR00T_PORT", "22"))
_DEFAULT_USER = os.environ.get("GR00T_USER", "root")
_DEFAULT_PASSWORD = os.environ.get("GR00T_PASSWORD", "")

HOST = _DEFAULT_HOST
PORT = _DEFAULT_PORT
USER = _DEFAULT_USER
PASSWORD = _DEFAULT_PASSWORD


def describe():
    return "%s@%s:%s" % (USER, HOST, PORT)

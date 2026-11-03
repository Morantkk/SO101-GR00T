"""SFTP 上传文件/目录（Windows 无 sshpass 时的替代方案）。

用法:
    <venv-python> upload_file.py <本地文件> <远程路径>
    <venv-python> upload_file.py --dir <本地目录> <远程目录>

实例: ssh -p 43912 root@connect.bjb1.seetacloud.com
"""
import os
import sys
import time

import paramiko

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from conn import HOST, PASSWORD, PORT, USER  # 换实例只改 conn.py
except ImportError:  # 兜底
    HOST = "connect.bjb1.seetacloud.com"
    PORT = 43912
    USER = "root"
    PASSWORD = "FKNErOycPYzN"


def sftp_connect(retries=5, delay=8):
    last = None
    for i in range(retries):
        try:
            transport = paramiko.Transport((HOST, PORT))
            transport.banner_timeout = 60
            transport.auth_timeout = 60
            transport.connect(username=USER, password=PASSWORD)
            return paramiko.SFTPClient.from_transport(transport), transport
        except Exception as e:  # noqa: BLE001
            last = e
            sys.stderr.write("[retry %d/%d] sftp connect failed: %r\n" % (i + 1, retries, e))
            time.sleep(delay)
    raise last


def ensure_remote_dir(sftp, remote_dir):
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.mkdir(cur)
        except IOError:
            pass


def put_dir(sftp, local_dir, remote_dir):
    ensure_remote_dir(sftp, remote_dir)
    n = 0
    for root, dirs, files in os.walk(local_dir):
        rel = os.path.relpath(root, local_dir)
        cur = remote_dir if rel == "." else remote_dir + "/" + rel.replace("\\", "/")
        ensure_remote_dir(sftp, cur)
        for f in files:
            lp = os.path.join(root, f)
            rp = cur + "/" + f
            sftp.put(lp, rp)
            n += 1
            print("  -> %s (%.1f MB)" % (rp, os.path.getsize(lp) / 1e6), flush=True)
    return n


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    sftp, transport = sftp_connect()
    try:
        if args[0] == "--dir":
            n = put_dir(sftp, args[1], args[2])
            print("[DONE] uploaded %d files" % n, flush=True)
        else:
            lp, rp = args[0], args[1]
            size = os.path.getsize(lp)
            print("uploading %s -> %s (%.1f MB) ..." % (lp, rp, size / 1e6), flush=True)
            t0 = time.time()
            sftp.put(lp, rp)
            dt = time.time() - t0
            # 校验远程文件大小必须用 sftp.stat()
            rsize = sftp.stat(rp).st_size
            print("[DONE] %.1f MB in %.1fs (%.1f MB/s), remote=%d bytes" % (
                size / 1e6, dt, size / 1e6 / max(dt, 1e-9), rsize), flush=True)
    finally:
        sftp.close()
        transport.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""并行上传（**按偏移直写，不落分片** + 断块重试 + 内容哈希校验）。

为什么不直接用 `parallel_upload.py`：
  1. 它先把所有分片写到 `<tmp>/_upload_parts` 再 `cat` 合成 ⇒ **峰值占 2 倍文件大小**。
     这次盘只剩几 GB，2×5GB 直接失败。
  2. 任一 block 失败只打印 "part XX FAILED" 然后继续 ⇒ 合成出来的文件大小可能对、**内容缺一块**。
     （本项目历史上就吃过"大小对、内容全 0"的亏。）
  3. 不校验内容。

本脚本：先 `truncate` 出目标大小的空文件，然后 N 条 SFTP 连接各自 `seek(offset)` 直写不同区间；
每个 block 最多重试 4 次（重连）；全部写完用**远端 sha256sum vs 本地 sha256** 逐文件核对。

用法：
  <venv-python> parallel_upload_checked.py <local_dir> <remote_dir> [threads]
例：
  <venv-python> parallel_upload_checked.py \
      "E:/trained_models/groot_v1" \
      /root/autodl-tmp/BACKUP_v1_model 10
"""
import argparse
import hashlib
import os
import posixpath
import socket
import sys
import threading
import time

import paramiko

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from conn import HOST, PASSWORD, PORT, USER
except ImportError:  # noqa: BLE001
    HOST, PORT, USER, PASSWORD = "connect.bjb2.seetacloud.com", 10607, "root", "0sJFrxolne7k"

CHUNK = 16 << 20          # 每个 block 16 MB
BIG = 8 << 20             # 小于这个的就整文件一条流


def open_sftp(tries=8, delay=4):
    """建 SFTP（带重试）。★ 用 socket 自己建 TCP 并设超时，避免 paramiko 内部无超时卡死。"""
    last = None
    for i in range(tries):
        try:
            s = socket.create_connection((HOST, PORT), timeout=25)
            s.settimeout(None)
            tr = paramiko.Transport(s)
            tr.banner_timeout = 45
            tr.auth_timeout = 45
            tr.set_keepalive(20)
            tr.connect(username=USER, password=PASSWORD)
            return paramiko.SFTPClient.from_transport(tr), tr
        except Exception as ex:  # noqa: BLE001
            last = ex
            sys.stderr.write("  [sftp %d/%d] %r\n" % (i + 1, tries, ex))
            time.sleep(delay)
    raise last


def sha256_local(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("local_dir")
    ap.add_argument("remote_dir")
    ap.add_argument("threads", nargs="?", type=int, default=10)
    o = ap.parse_args()

    files = []
    for root, _, names in os.walk(o.local_dir):
        for n in names:
            lp = os.path.join(root, n)
            rel = os.path.relpath(lp, o.local_dir).replace("\\", "/")
            files.append((lp, rel, os.path.getsize(lp)))
    files.sort(key=lambda x: -x[2])
    if not files:
        print("本地目录是空的：%s" % o.local_dir)
        return 1
    total = sum(f[2] for f in files)
    print("待上传 %d 个文件，共 %.2f GB（%d 并发，16MB/块，直写不落分片）"
          % (len(files), total / 1e9, o.threads))

    sf, tr = open_sftp()
    try:
        def _mkdirs(path):
            """逐级创建（sftp 没有 makedirs）。已存在的层级跳过，避免每个文件都重走一遍（否则 311 个文件要 1500 次往返）。"""
            parts = path.strip("/").split("/")
            cur = ""
            for p in parts:
                cur += "/" + p
                if cur in _made:
                    continue
                try:
                    sf.stat(cur)
                except IOError:
                    try:
                        sf.mkdir(cur)
                    except IOError:
                        pass
                _made.add(cur)

        _made = set()
        dirs = sorted({posixpath.dirname(posixpath.join(o.remote_dir, rel)) for _, rel, _ in files})
        for d in dirs:
            _mkdirs(d)
        print("远端目录就绪：%d 个" % len(dirs), flush=True)
    finally:
        try:
            tr.close()
        except Exception:  # noqa: BLE001
            pass

    t_all = time.time()
    for lp, rel, size in files:
        rp = posixpath.join(o.remote_dir, rel)
        print("\n--- %s  (%.1f MB)" % (rel, size / 1e6), flush=True)
        # 预分配（truncate 出目标大小）
        sf, tr = open_sftp()
        try:
            with sf.open(rp, "wb") as f:
                f.truncate(size)
        finally:
            tr.close()

        if size <= BIG:
            tasks = [(0, size)]
        else:
            tasks = [(s, min(s + CHUNK, size)) for s in range(0, size, CHUNK)]
        todo = sum(e - s for s, e in tasks)
        print("   切成 %d 块" % len(tasks), flush=True)

        lock = threading.Lock()
        done_bytes = [0]
        nblk = [0]
        failed = []
        t0 = time.time()

        def worker():
            try:
                sf2, tr2 = open_sftp()
            except Exception as ex:  # noqa: BLE001
                with lock:
                    failed.append(("connect", repr(ex)))
                return
            local = open(lp, "rb")
            try:
                while True:
                    with lock:
                        if not tasks:
                            break
                        s, e = tasks.pop()
                    for attempt in range(4):
                        try:
                            local.seek(s)
                            buf = local.read(e - s)
                            if len(buf) != e - s:
                                raise IOError("本地短读 %d/%d" % (len(buf), e - s))
                            with sf2.open(rp, "r+b") as f:
                                f.seek(s)
                                f.write(buf)
                            with lock:
                                done_bytes[0] += len(buf)
                                nblk[0] += 1
                                if nblk[0] % 8 == 0 or done_bytes[0] >= todo:
                                    el = max(time.time() - t0, 1e-6)
                                    print("   %6.0f/%6.0f MB  %.1f MB/s"
                                          % (done_bytes[0] / 1e6, todo / 1e6,
                                             done_bytes[0] / el / 1e6), flush=True)
                            break
                        except Exception as ex:  # noqa: BLE001
                            if attempt == 3:
                                with lock:
                                    failed.append(("%d-%d" % (s, e), repr(ex)))
                            else:
                                time.sleep(2 * (attempt + 1))
            finally:
                local.close()
                try:
                    sf2.close()
                    tr2.close()
                except Exception:  # noqa: BLE001
                    pass

        ths = [threading.Thread(target=worker) for _ in range(min(o.threads, len(tasks)))]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        if failed:
            print("   !! %d 个 block 失败：%s" % (len(failed), failed[:3]))
            return 2
        print("   上传完成 %.1fs（%.2f MB/s）" % (time.time() - t0, size / 1e6 / max(time.time() - t0, 1e-6)))

    # ---- 内容校验：远端 sha256 vs 本地 sha256 ----
    print("\n================ 内容校验（sha256）================")
    local_hash = {rel: sha256_local(lp) for lp, rel, _ in files}
    sf, tr = open_sftp()
    bad = 0
    try:
        for lp, rel, size in files:
            rp = posixpath.join(o.remote_dir, rel)
            cmd = "sha256sum %s" % rp
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=60)
            _, so, _ = cli.exec_command(cmd, timeout=1800)
            got = so.read().decode().split()[0] if so else ""
            cli.close()
            ok = (got == local_hash[rel])
            if not ok:
                bad += 1
            print("  %-46s %12d B  %s" % (rel, size, "OK" if ok else "FAIL 远端=%s" % got[:16]))
    finally:
        try:
            tr.close()
        except Exception:  # noqa: BLE001
            pass
    print("\n%s（总用时 %.1f min）"
          % ("全部通过 ✓" if bad == 0 else "有 %d 个不一致 ✗" % bad, (time.time() - t_all) / 60))
    return 0 if bad == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())

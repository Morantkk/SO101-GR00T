#!/usr/bin/env python3
"""通用「并行目录下载器」：把远端一个目录整棵拉回本地。

为什么需要：单条 paramiko SFTP 只有 0.5~1.3 MB/s，7 GB 的模型要跑一两个小时。
做法：把每个文件切成若干**字节区间**，用 N 条并发连接同时拉，各自直接写进本地文件的对应偏移。

特点：
  · 自动递归远端目录
  · 大文件按区间并发；小文件整体并发
  · ★ **校验以内容为准**：远端 `sha256sum` + 本地 sha256 逐文件比对（不再只比字节数）
  · ★ **幂等靠哈希**：只有"本地 sha256 == 远端 sha256"才跳过；大小对但内容错的残件会被删掉重下
  · ★ **失败即清残件**：任一块下载失败 → 删除该文件的本地残件，杜绝"大小对、内容全零"的假文件
  · 先预分配本地文件再按 offset 写，避免拼接

⚠️ 历史事故（2026-10-15，务必记住）：
  早期版本只比**字节数**校验，且用 `truncate()` 预分配文件 → 传输中途一旦断开，
  本地会留下"大小完全正确、内容全是 0"的文件；重跑时又被"大小相同就跳过"命中，最后校验**还报 OK**。
  后果：`trained_models/groot_arm101_pick_one_pen_v1` 的 7.5 GB 全零（只有 index.json 完好），
  而远端副本已被删除 ⇒ **v1 模型实际丢失**。
  ⇒ 结论：**任何下载都要用内容哈希校验，永远不要只比大小。**

用法：
  <venv-python> parallel_download.py <remote_dir> <local_dir> [threads] [--size-only] [--hash-file F]
例：
  <venv-python> parallel_download.py /root/BACKUP_gr00t_official/model \\
      "E:/trained_models/groot_v1" 12

★ 2026-10-15 补充（弱网/高负载实例）：
  在 CPU 被超卖（load 30+）或被限速的实例上，让本脚本**远程跑 sha256sum** 会长时间无响应，
  且 SSH 长会话容易被网关掐掉（报 "Error reading SSH protocol banner"）。稳妥做法：
    ① 远端后台自己算好哈希：  cd <remote_dir> && setsid nohup bash -c 'sha256sum ./* > /root/xxx_sha256.txt' &
    ② 拉回该文件，再用 --hash-file 跑本脚本 —— 跳过远程哈希这一步。
  另外 open_sftp / 下载块都加了重试，单块失败会重排队而不是直接放弃。
"""
import argparse
import hashlib
import os
import posixpath
import socket
import stat
import sys
import threading
import time

import paramiko

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:  # 连接信息统一在 conn.py，换实例只改那一处
    from conn import HOST, PASSWORD, PORT, USER
except ImportError:
    HOST, PORT, USER, PASSWORD = "connect.bjb1.seetacloud.com", 43912, "root", "FKNErOycPYzN"

CHUNK = 16 << 20      # 大文件每个区间 16 MB（★ 固定块，别按线程数分整文件）
SMALL = 8 << 20       # 小于这个就整体下，不切
HBUF = 8 << 20        # 哈希读取缓冲
BATCH = 100           # 远端 sha256sum 每次最多传多少个文件


def _q(s):
    """单引号包住，内部单引号转义 —— 远端 shell 安全。"""
    return "'" + str(s).replace("'", "'\\''") + "'"


def open_sftp(tries=10, delay=6):
    """建立 SFTP 连接（带重试；弱网/高负载实例上一次连不上是常态）。

    ★ 两个坑（2026-10-15 实测 AutoDL 无卡/高负载实例）：
      ① 高负载实例（load 30+）上 sshd 会被饿死 → 常报
         "Error reading SSH protocol banner"。所以超时别设太长，宁可快速失败多试几次。
      ② `paramiko.Transport((host, port))` 内部做 TCP connect **不带超时**，
         网关半死不活时会把整个进程卡死几分钟（日志里一个字都不出）。
         ⇒ 必须自己用 socket.create_connection(..., timeout=N) 建连再交给 Transport。
    """
    last = None
    for i in range(tries):
        try:
            sock = socket.create_connection((HOST, PORT), timeout=20)
            sock.settimeout(None)          # 交给 paramiko 后恢复阻塞模式
            tr = paramiko.Transport(sock)
            tr.banner_timeout = 45
            tr.auth_timeout = 45
            tr.set_keepalive(20)
            tr.connect(username=USER, password=PASSWORD)
            return paramiko.SFTPClient.from_transport(tr), tr
        except Exception as ex:  # noqa: BLE001
            last = ex
            sys.stderr.write("[sftp %d/%d] %r\n" % (i + 1, tries, ex))
            sys.stderr.flush()
            time.sleep(delay)
    raise last


def walk(sftp, root):
    files, stack = [], [""]
    while stack:
        rel = stack.pop()
        rp = posixpath.join(root, rel) if rel else root
        for a in sftp.listdir_attr(rp):
            r = posixpath.join(rel, a.filename) if rel else a.filename
            if stat.S_ISDIR(a.st_mode):
                stack.append(r)
            elif stat.S_ISREG(a.st_mode):
                files.append((r, a.st_size))
    return sorted(files)


def sha256_local(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(HBUF)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_remote(remote_dir, rels, tries=5):
    """一次 SSH 会话批量取远端 sha256 → {rel: hex}。

    ★ 弱网/高负载实例上 `sha256sum` 可能很慢或会话被掐断：
      逐行读、失败自动重连重试（已成功的批次不重算）。
    """
    out = {}
    todo = list(rels)
    attempt = 0
    while todo and attempt < tries:
        attempt += 1
        cli = paramiko.SSHClient()
        cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            cli.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                        timeout=60, banner_timeout=90, auth_timeout=90,
                        allow_agent=False, look_for_keys=False)
            tr = cli.get_transport()
            if tr is not None:
                tr.set_keepalive(20)
            still = []
            for i in range(0, len(todo), BATCH):
                batch = todo[i:i + BATCH]
                cmd = "cd %s && sha256sum %s" % (
                    _q(remote_dir), " ".join(_q(r) for r in batch))
                _, so, se = cli.exec_command(cmd, timeout=7200)
                ch = so.channel
                got = 0
                for line in iter(so.readline, ""):
                    h, _, p = line.rstrip("\n").partition("  ")
                    if h and p and len(h) == 64:
                        out[p.strip()] = h.strip()
                        got += 1
                rc = ch.recv_exit_status()
                if rc != 0 or got < len(batch):
                    # 这一批没拿全 → 记下来，下一轮重试
                    for r in batch:
                        if r not in out:
                            still.append(r)
                    if rc != 0:
                        sys.stderr.write("[hash] rc=%s: %s\n" % (
                            rc, se.read().decode("utf-8", "replace")[:200]))
                else:
                    print("   哈希 %d/%d" % (len(out), len(rels)), flush=True)
            todo = still
        except Exception as ex:  # noqa: BLE001
            sys.stderr.write("[hash retry %d/%d] %r\n" % (attempt, tries, ex))
            time.sleep(10)
        finally:
            try:
                cli.close()
            except Exception:  # noqa: BLE001
                pass
    if todo:
        raise RuntimeError("远端 sha256 仍有 %d 个文件没拿到: %s" % (len(todo), todo[:5]))
    return out


def load_hash_file(path, remote_dir):
    """从本地文件读 `sha256sum` 输出（形如 "<hex>  <name>"）。"""
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            h, _, p = line.rstrip("\n").partition("  ")
            p = p.strip()
            if p.startswith("./"):      # sha256sum ./* 的输出带 ./ 前缀
                p = p[2:]
            if not h or len(h) != 64 or not p or p == "/dev/null":
                continue
            # sha256sum 的输出是相对 remote_dir 的（cd 过），也可能是完整路径
            if p.startswith("/"):
                p = posixpath.relpath(p, remote_dir)
            out[p] = h
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("remote_dir")
    ap.add_argument("local_dir")
    ap.add_argument("threads", nargs="?", type=int, default=8)
    ap.add_argument("--size-only", action="store_true",
                    help="只比字节数（快，但**不可靠**，仅调试用；正常别加）")
    ap.add_argument("--hash-file", default=None,
                    help="从本地文件读远端 sha256sum 输出，跳过「远程算哈希」这一步（弱网实例推荐）")
    o = ap.parse_args()

    sf, tr = open_sftp()
    try:
        files = walk(sf, o.remote_dir)
    finally:
        try:
            tr.close()
        except Exception:  # noqa: BLE001
            pass
    if not files:
        print("!! 远端目录是空的或不存在:", o.remote_dir)
        return 1
    grand = sum(s for _, s in files)
    print("远端 %s：%d 个文件，共 %.1f MB" % (o.remote_dir, len(files), grand / 1e6), flush=True)

    rels = [r for r, _ in files]
    rhash = {}
    if o.hash_file:
        rhash = load_hash_file(o.hash_file, o.remote_dir)
        miss = [r for r in rels if r not in rhash]
        print("从哈希文件读到 %d/%d 个（缺 %d 个）" % (len(rhash), len(rels), len(miss)))
        if miss:
            print("   缺失的先用远程算：", miss[:5], flush=True)
            rhash.update(sha256_remote(o.remote_dir, miss))
    elif not o.size_only:
        print("取远端 sha256（内容校验用，大文件要几十秒）…", flush=True)
        t = time.time()
        rhash = sha256_remote(o.remote_dir, rels)
        print("   远端 sha256 就绪：%d/%d 个，用时 %.1fs" % (len(rhash), len(rels), time.time() - t))

    tasks, skipped = [], 0
    for rel, size in files:
        lp = os.path.join(o.local_dir, rel.replace("/", os.sep))
        d = os.path.dirname(lp)
        if d:
            os.makedirs(d, exist_ok=True)
        ok = os.path.exists(lp) and os.path.getsize(lp) == size
        if ok and not o.size_only:
            ok = sha256_local(lp) == rhash.get(rel)
        if ok:
            skipped += 1
            print("  跳过（内容一致） %s  %.1f MB" % (rel, size / 1e6))
            continue
        if os.path.exists(lp):
            print("  重下（本地内容与远端不一致） %s" % rel)
            try:
                os.remove(lp)      # ★ 残件 / 大小对但内容错 → 删掉
            except OSError as ex:  # 某些环境（沙箱/占用）删不掉就直接覆盖
                print("     （删不掉，直接覆盖：%s）" % ex)
        if size == 0:
            open(lp, "wb").close()
            continue
        with open(lp, "wb") as f:
            f.truncate(size)
        if size <= SMALL:
            tasks.append((posixpath.join(o.remote_dir, rel), lp, 0, size))
        else:
            # ★ 固定块大小（CHUNK），块数 = ceil(size/CHUNK)。
            #   绝不要写成 step = size/threads —— 那样 14 线程会把 5GB 切成 14 个 ~340MB 的巨块：
            #   ① 显存/内存里要攒 340MB 才落盘 → 慢网下一个块要十几分钟、"零进度"看着像卡死；
            #   ② 负载不均（谁摊到大块谁最慢）；③ 中断时整块白读。
            #   固定小块还能让"落盘进度"随时可见（可抽样看非零字节）。
            for s in range(0, size, CHUNK):
                tasks.append((posixpath.join(o.remote_dir, rel), lp, s, min(s + CHUNK, size)))

    todo = sum(e - s for _, _, s, e in tasks)
    print("待下载 %.1f MB，切成 %d 个块，用 %d 条并发连接"
          % (todo / 1e6, len(tasks), o.threads), flush=True)

    lock = threading.Lock()
    st = {"bytes": 0, "blocks": 0}
    failed = []                # (rp, lp, s, e, err)
    total_blocks = len(tasks)
    t0 = time.time()
    RETRY = 4                  # 单块最多重试次数（弱网实例必备）

    def worker():
        try:
            sf2, tr2 = open_sftp()
        except Exception as ex:  # noqa: BLE001
            with lock:
                failed.append(("<connect>", "", 0, 0, repr(ex)))
            return
        try:
            while True:
                with lock:
                    if not tasks:
                        break
                    rp, lp, s, e = tasks.pop()
                last = None
                for attempt in range(RETRY + 1):
                    try:
                        with sf2.open(rp, "rb") as f:
                            f.seek(s)
                            remain, buf = e - s, bytearray()
                            while remain > 0:
                                b = f.read(min(4 << 20, remain))
                                if not b:
                                    break
                                buf += b
                                remain -= len(b)
                        if len(buf) != e - s:
                            raise IOError("短读 got=%d want=%d" % (len(buf), e - s))
                        with open(lp, "r+b") as g:
                            g.seek(s)
                            g.write(bytes(buf))
                        last = None
                        break
                    except Exception as ex:  # noqa: BLE001
                        last = ex
                        # 会话可能已被网关掐掉 → 重连再试（每次只重试失败的那一块）
                        try:
                            tr2.close()
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            sf2, tr2 = open_sftp(tries=2, delay=3)
                        except Exception:  # noqa: BLE001
                            time.sleep(5)
                if last is not None:
                    with lock:
                        failed.append((rp, lp, s, e, repr(last)))
                else:
                    with lock:
                        st["bytes"] += (e - s)
                        st["blocks"] += 1
                        if st["blocks"] % 3 == 0 or st["blocks"] <= 2 or st["blocks"] == total_blocks:
                            el = max(time.time() - t0, 1e-6)
                            print("   %6.1f/%6.1f MB   %5.2f MB/s   (%d/%d 块)"
                                  % (st["bytes"] / 1e6, todo / 1e6, st["bytes"] / el / 1e6,
                                     st["blocks"], total_blocks), flush=True)
        finally:
            try:
                tr2.close()
            except Exception:  # noqa: BLE001
                pass

    ths = [threading.Thread(target=worker, daemon=True) for _ in range(o.threads)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()

    el = max(time.time() - t0, 1e-6)
    print("\n下载 %.1f MB，用时 %.1f s（平均 %.2f MB/s）"
          % (st["bytes"] / 1e6, el, st["bytes"] / el / 1e6))

    # ★ 任一块失败 → 删除该文件残件，避免下次被"大小相同"骗过
    if failed:
        dead = sorted({lp for _, lp, _, _, _ in failed if lp})
        for lp in dead:
            try:
                os.remove(lp)
            except OSError:
                pass
        print("!! %d 个块失败；已删除 %d 个残件（重跑本脚本会自动补下）："
              % (len(failed), len(dead)))
        for f in failed[:10]:
            print("   ", f)

    # ---------------- 校验（内容优先） ----------------
    print("--- 校验（%s）---" % ("字节数" if o.size_only else "sha256 内容"))
    bad = 0
    for rel, size in files:
        lp = os.path.join(o.local_dir, rel.replace("/", os.sep))
        if not os.path.exists(lp):
            print("   %-52s %12d vs %-12s  !! 缺失" % (rel, size, "-"))
            bad += 1
            continue
        got = os.path.getsize(lp)
        if got != size:
            print("   %-52s %12d vs %12d  !! 大小不一致" % (rel, size, got))
            bad += 1
            continue
        if o.size_only:
            print("   %-52s %12d vs %12d  OK(仅大小)" % (rel, size, got))
            continue
        gh = sha256_local(lp)
        want = rhash.get(rel)
        okk = (want is not None and gh == want)
        if not okk:
            bad += 1
        print("   %-52s %12d   sha %s  %s" % (rel, size, gh[:12],
                                              "OK" if okk else "!! 内容不一致 (远端 %s)" % (want or "?")[:12]))

    ok_all = (bad == 0 and not failed)
    print("\n%s  ->  %s" % ("全部校验通过 ✓" if ok_all else "有失败/不一致 ✗", o.local_dir))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

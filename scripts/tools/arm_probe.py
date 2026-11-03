#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""后台采样两个 CH343 口的位置，用来判定哪个是主臂：手动转动主臂 → 变化大的那个口就是主臂。

用法： python arm_probe.py 90        # 采样 90 秒，写 outputs/arm_probe.csv
"""
import sys
import time

import serial.tools.list_ports as lp
from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS

SECS = int(sys.argv[1]) if len(sys.argv) > 1 else 90
BAUD = 1_000_000

ports = [p.device for p in lp.comports() if "CH34" in (p.description or "") or "1A86" in (p.hwid or "")]
handles = {}
for dev in ports:
    for proto in (0, 1):
        ph = PortHandler(dev)
        if not ph.openPort():
            break
        ph.setBaudRate(BAUD)
        pk = PacketHandler(proto)
        _, c, _ = pk.ping(ph, 1)
        if c == COMM_SUCCESS:
            handles[dev] = (ph, pk)
            break
        ph.closePort()

print("采样端口:", list(handles), flush=True)
rows = []
t0 = time.time()
while time.time() - t0 < SECS:
    rec = [round(time.time() - t0, 2)]
    for dev in ports:
        if dev in handles:
            ph, pk = handles[dev]
            for mid in range(1, 7):
                pos, c, _ = pk.read2ByteTxRx(ph, mid, 56)
                rec.append(pos if c == COMM_SUCCESS else -1)
        else:
            rec += [-1] * 6
    rows.append(rec)
    time.sleep(0.12)

for ph, _ in handles.values():
    ph.closePort()

hdr = "t," + ",".join("%s_id%d" % (d, i) for d in ports for i in range(1, 7))
with open("outputs/arm_probe.csv", "w", encoding="utf-8") as f:
    f.write(hdr + "\n")
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")

print("采样 %d 行 -> outputs/arm_probe.csv" % len(rows), flush=True)
import numpy as np

A = np.array([r[1:] for r in rows], dtype=float)
k = 0
for d in ports:
    seg = A[:, k:k + 6]
    rng = np.nanmax(seg, axis=0) - np.nanmin(seg, axis=0)
    print("%s  各轴角度变化量: %s   总变化=%.0f" % (d, np.round(rng, 0).astype(int).tolist(), rng.sum()))
    k += 6

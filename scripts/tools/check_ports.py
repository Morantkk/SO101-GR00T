#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查端口：相机索引 ↔ 设备名 ↔ 分辨率（+ 存图），串口 ↔ 舵机是否在线 ↔ 电压。

只读，不给舵机发任何动作命令。
用法： python check_ports.py [--grab]
"""
import sys

import cv2

PICS = ["cam0.png", "cam1.png", "cam2.png"]


def cams(grab=True):
    from pygrabber.dshow_graph import FilterGraph
    names = FilterGraph().get_input_devices()
    print("=== 相机（DirectShow 顺序 = cv2.CAP_DSHOW 的 index）===")
    for i, nm in enumerate(names):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("  %d | %-22s | 打不开" % (i, nm))
            continue
        ok, fr = False, None
        for _ in range(20):
            ok, fr = cap.read()
            if ok:
                break
        cap.release()
        if not ok:
            print("  %d | %-22s | 有设备但读不到帧" % (i, nm))
            continue
        h, w = fr.shape[:2]
        print("  %d | %-22s | 帧 %dx%d" % (i, nm, w, h))
        if grab:
            cv2.putText(fr, "index %d  (%s)  %dx%d" % (i, nm[:16], w, h),
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            ok2, buf = cv2.imencode(".png", fr)
            if ok2:
                buf.tofile(PICS[i])          # Windows 上含中文路径 imwrite 会静默失败
                print("        -> 存图 %s" % PICS[i])


def arms():
    import serial.tools.list_ports as lp
    from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS
    print("\n=== 串口 ===")
    ports = [p for p in lp.comports() if "CH34" in (p.description or "") or "1A86" in (p.hwid or "")]
    for p in lp.comports():
        tag = "臂" if p in ports else "跳过"
        print("  %-6s | %-38s | %s | SER=%s" % (p.device, p.description, tag,
                                                (p.hwid or "").split("SER=")[-1][:12]))
    print("\n=== 舵机在线情况（只读寄存器）===")
    for p in ports:
        got = None
        for proto in (0, 1):
            ph = PortHandler(p.device)
            if not ph.openPort():
                print("  %s: 打不开（被占用？）" % p.device)
                break
            ph.setBaudRate(1000000)
            pk = PacketHandler(proto)
            m, c, e = pk.ping(ph, 1)
            if c == COMM_SUCCESS:
                got = (ph, pk, proto)
                break
            ph.closePort()
        if not got:
            print("  %s: ⚠ 1 号舵机 ping 不通" % p.device)
            continue
        ph, pk, proto = got
        print("  %s: protocol=%d 在线" % (p.device, proto))
        for mid in range(1, 7):
            pos, c1, _ = pk.read2ByteTxRx(ph, mid, 56)
            vol, c2, _ = pk.read1ByteTxRx(ph, mid, 62)
            tmp, c3, _ = pk.read1ByteTxRx(ph, mid, 63)
            tq, c4, _ = pk.read1ByteTxRx(ph, mid, 40)
            if c1 != COMM_SUCCESS:
                print("     id%d 无响应" % mid)
                continue
            print("     id%d 位置=%5d  电压=%4.1fV  温度=%2d℃  torque=%d"
                  % (mid, pos, vol / 10.0, tmp, tq))
        ph.closePort()


if __name__ == "__main__":
    cams(grab="--grab" in sys.argv)
    arms()

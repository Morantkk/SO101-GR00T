#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""官方 Isaac-GR00T 策略服务 = 官方 `inference_service.py --server` + **JPEG 解码补丁**。

为什么需要
    官方服务端只接受**原始 uint8 图像数组**。两路 640×360 原图 = 691KB × 2 ≈ **1.38MB/次**；
    经 paramiko 隧道（实测 ~1.3MB/s）单次要 ~1s → 控制环被上行带宽卡死。
    把图像压成 JPEG 再传（q90 通常 20~50KB/路）⇒ 上行降到 ~50KB，往返降到 ~0.15s。

★ 只改**传输**，不改模型看到的东西
    本服务把 JPEG 解回 (H,W,3) uint8 RGB 再补一维成 (1,H,W,3)，与"客户端直接发原图"的语义**完全一致**；
    之后照走官方 `Gr00tPolicy` 的原样变换（VideoCrop(0.95) → Resize(224) → 归一化）。
    训练数据本身就是 H.264 压缩过的，JPEG q90（PSNR≈40dB）带来的分布差异可忽略。

线协议（客户端 → 本服务）
    {"video.front": {"__jpeg__": True, "bytes": <jpeg>}, ...}
    · 值做成这个 dict 就是 JPEG；直接发 ndarray 则原样透传（等同官方行为）。
    · **按键无关**解包（不匹配 `video.` 前缀），避免以后换键名就崩。

用法
    python 100_policy_server_jpeg.py --model-path /root/BACKUP_gr00t_official/model --port 5555
    可选：--no-jpeg / --embodiment-tag / --data-config / --denoising-steps / --host / --api-token
"""
import argparse

import numpy as np


class JpegDecodingPolicy:
    """给 Gr00tPolicy 包一层：把 {"__jpeg__":True,"bytes":...} 解成 (1,H,W,3) uint8 RGB。"""

    def __init__(self, policy, enabled=True, verbose_every=50):
        self._policy = policy
        self.enabled = enabled
        self.verbose_every = verbose_every
        self.n_decoded = 0
        self.bytes_in = 0

    def __getattr__(self, name):
        # 只转发未知属性（get_modality_config 等）；dunder 不转发，免得出怪问题
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._policy, name)

    def _decode_one(self, v):
        import cv2

        if isinstance(v, dict) and v.get("__jpeg__"):
            raw = np.frombuffer(v["bytes"], dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)          # -> BGR
            if img is None:
                raise ValueError("JPEG 解码失败（收到 %d 字节）" % len(raw))
            self.n_decoded += 1
            self.bytes_in += len(raw)
            if self.verbose_every and self.n_decoded % self.verbose_every == 1:
                print("[jpeg] #%d  收到 %d 字节 -> %dx%d  平均 %.1fKB/帧"
                      % (self.n_decoded, len(raw), img.shape[1], img.shape[0],
                         self.bytes_in / self.n_decoded / 1024.0), flush=True)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[None, ...]   # ★ 补 batch 维，与发原图一致
        return v

    def get_action(self, obs):
        if self.enabled:
            obs = {k: self._decode_one(v) for k, v in obs.items()}
        return self._policy.get_action(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--data-config", default="so100_dualcam")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--api-token", default=None)
    ap.add_argument("--no-jpeg", action="store_true", help="不解 JPEG（等同官方行为）")
    a = ap.parse_args()

    from gr00t.eval.robot import RobotInferenceServer
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    dcfg = load_data_config(a.data_config)
    policy = Gr00tPolicy(
        model_path=a.model_path,
        modality_config=dcfg.modality_config(),
        modality_transform=dcfg.transform(),
        embodiment_tag=a.embodiment_tag,
        denoising_steps=a.denoising_steps,
    )
    wrapped = JpegDecodingPolicy(policy, enabled=not a.no_jpeg)
    print("[server] jpeg_decode=%s  model=%s  data_config=%s  denoise=%d  port=%d"
          % (not a.no_jpeg, a.model_path, a.data_config, a.denoising_steps, a.port), flush=True)
    RobotInferenceServer(wrapped, host=a.host, port=a.port, api_token=a.api_token).run()


if __name__ == "__main__":
    main()

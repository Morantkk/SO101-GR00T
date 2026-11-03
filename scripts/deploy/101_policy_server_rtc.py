#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""官方 Isaac-GR00T 策略服务 + JPEG 解码 + **RTC（最小可用版：硬前缀 inpainting）**。

与 `100_policy_server_jpeg.py` 的唯一区别
    在动作头的采样循环里，把「上一块已经被执行掉的前 d 步」钉死在那条旧轨迹上，
    让新块的剩余步**以这条真实执行过的前缀为条件**生成 ⇒ 块切换处不再跳变。

为什么最小版就够了
    GR00T 的 DiT 对全部 action token（horizon=16）做**全自注意力**，
    所以把前 d 步硬钉住之后，后面的步会自然被"拉"到与它连续的轨迹上——
    这正是 RTC 论文里 inpainting 起作用的原因，不需要额外的 guidance 项。

线协议（在 100 的基础上多两个可选键，服务端会 pop 掉再交给 Gr00tPolicy）
    "__rtc.d"     : int   上一块客户端**实际执行**了多少步（缺失 / 0 = 不做约束）
    "__rtc.reset" : bool  复位 / 换场景时置 True ⇒ 清空前缀（避免拿旧场景的轨迹去钉）

注入点
    gr00t/model/action_head/flow_matching_action_head.py::FlowMatchingActionHead.get_action
    原循环：x0 = randn → for t: v = model(x_t) → x ← x + dt·v
    本补丁重写同一循环，并对前 d 步：
        v[:, :d] = prefix[:, :d] - x0[:, :d]              # flow matching 直线路径上的精确速度
        x[:, :d] = (1-tc)·x0[:, :d] + tc·prefix[:, :d]    # 强制回到 noise→prefix 的直线上
    前缀取自**上一次本层输出的归一化动作**（在 policy 反归一化之前），所以量纲天然一致。

用法
    python 101_policy_server_rtc.py --model-path /root/autodl-tmp/gr00t_trim2 --port 5555
    可选：--rtc-extra 1（把客户端上报的 d 再加几步，补偿往返延迟）
          --no-rtc（等价于 100_policy_server_jpeg.py，用于 A/B）
          --no-jpeg / --data-config / --embodiment-tag / --denoising-steps / --host
"""
import argparse
from types import MethodType

import numpy as np

# 采样补丁与 policy 包装共享的状态
_RTC = {"prefix": None, "last_d": 0, "on": True, "extra": 0, "n_calls": 0, "n_pinned": 0,
        "beta": 1.0, "decay": True}


# ────────────────────────────────────────────────────────── 动作头采样补丁
def install_rtc_patch(policy):
    """把 FlowMatchingActionHead.get_action 换成带前缀约束的版本。"""
    import torch
    from transformers.feature_extraction_utils import BatchFeature

    head = policy.model.action_head
    H = int(head.config.action_horizon)
    D = int(head.config.action_dim)

    def rtc_get_action(self, backbone_output, action_input):
        with torch.no_grad():
            backbone_output = self.process_backbone_output(backbone_output)
            vl_embs = backbone_output.backbone_features
            embodiment_id = action_input.embodiment_id
            state_features = self.state_encoder(action_input.state, embodiment_id)

            batch_size = vl_embs.shape[0]
            device = vl_embs.device
            # ★ 初始噪声必须留住：前缀约束是"沿 noise→prefix 这条直线"施加的
            x0 = torch.randn(
                size=(batch_size, self.config.action_horizon, self.config.action_dim),
                dtype=vl_embs.dtype,
                device=device,
            )
            actions = x0.clone()

            prefix = _RTC["prefix"]
            d = 0
            guide = None          # 重叠区：旧计划剩下的 (H-d) 步
            beta_vec = None       # 逐 token 的引导权重
            if _RTC["on"] and prefix is not None:
                d = int(min(_RTC["last_d"], self.config.action_horizon))
                if prefix.shape[1] == self.config.action_horizon and 0 < d < self.config.action_horizon:
                    # ★ 关键：新块是**从当前时刻 t = t0+d 往后**的 16 步，
                    #   所以要和旧计划对齐的，是旧计划**剩下**的那段 prefix[:, d:]，不是前 d 步。
                    #   锁定"已经执行过的前 d 步"没有意义 —— 客户端根本不会再用它们，
                    #   等于白扔 d 个 token，剩下的自由发挥 ⇒ 接缝照样跳。
                    guide = prefix[:, d:].to(device=device, dtype=vl_embs.dtype)
                    n_g = guide.shape[1]
                    # ★ 论文原版的 RTC：**软引导**（不是硬钉死）
                    #   权重沿重叠区线性衰减：接缝处 β（强，保证连续）→ 末端 0（放开，让新观测说话）
                    if _RTC["decay"]:
                        beta_vec = _RTC["beta"] * torch.linspace(
                            1.0, 0.0, n_g, device=device, dtype=vl_embs.dtype)
                    else:
                        beta_vec = torch.full((n_g,), _RTC["beta"],
                                              device=device, dtype=vl_embs.dtype)
                    beta_vec = beta_vec.view(1, n_g, 1)      # ★ 广播成 (B, n, D)
                else:
                    d = 0

            num_steps = self.num_inference_timesteps
            dt = 1.0 / num_steps

            for t in range(num_steps):
                t_cont = t / float(num_steps)
                t_discretized = int(t_cont * self.num_timestep_buckets)

                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
                if self.config.add_pos_embed:
                    pos_ids = torch.arange(
                        action_features.shape[1], dtype=torch.long, device=device
                    )
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
                    vl_embs.shape[0], -1, -1
                )
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )
                pred = self.action_decoder(model_output, embodiment_id)
                pred_velocity = pred[:, -self.action_horizon :]

                if beta_vec is not None:
                    n = guide.shape[1]
                    # 论文版引导：v ← (1-β)·v_model + β·v_prefix
                    #   v_prefix = (prefix − x_t) / (1 − t)，即 flow matching 直线路径上
                    #   "指向旧计划终点"的精确速度；β 逐 token 衰减 ⇒ 接缝硬、末端软。
                    v_prefix = (guide - actions[:, :n]) / max(1.0 - t_cont, 0.05)
                    pred_velocity = pred_velocity.clone()
                    pred_velocity[:, :n] = (
                        (1.0 - beta_vec) * pred_velocity[:, :n] + beta_vec * v_prefix
                    )

                actions = actions + dt * pred_velocity

            _RTC["prefix"] = actions.detach().clone()
            _RTC["n_calls"] += 1
            if beta_vec is not None:
                _RTC["n_pinned"] += 1
            return BatchFeature(data={"action_pred": actions})

    head.get_action = MethodType(rtc_get_action, head)
    print("[rtc] 已接管 action_head.get_action  H=%d D=%d" % (H, D), flush=True)


# ────────────────────────────────────────────────────────── policy 包装
class RTCJpegPolicy:
    """JPEG 解码 + RTC 控制键剥离 + 状态重置。"""

    def __init__(self, policy, jpeg=True, verbose_every=50, log_every=1):
        self._policy = policy
        self.jpeg = jpeg
        self.verbose_every = verbose_every
        self.log_every = log_every
        self.n_decoded = 0
        self.bytes_in = 0
        self.n_req = 0

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._policy, name)

    def _decode_one(self, v):
        import cv2

        if isinstance(v, dict) and v.get("__jpeg__"):
            raw = np.frombuffer(v["bytes"], dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)  # -> BGR
            if img is None:
                raise ValueError("JPEG 解码失败（收到 %d 字节）" % len(raw))
            self.n_decoded += 1
            self.bytes_in += len(raw)
            if self.verbose_every and self.n_decoded % self.verbose_every == 1:
                print(
                    "[jpeg] #%d  收到 %d 字节 -> %dx%d  平均 %.1fKB/帧"
                    % (
                        self.n_decoded,
                        len(raw),
                        img.shape[1],
                        img.shape[0],
                        self.bytes_in / self.n_decoded / 1024.0,
                    ),
                    flush=True,
                )
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[None, ...]
        return v

    def get_action(self, obs):
        obs = dict(obs)
        d = obs.pop("__rtc.d", None)
        reset = bool(obs.pop("__rtc.reset", False))
        self.n_req += 1

        if reset or d is None:
            _RTC["prefix"] = None
            _RTC["last_d"] = 0
            tag = "reset" if reset else "无 d"
        else:
            _RTC["last_d"] = int(d) + int(_RTC["extra"])
            tag = "d=%s +%d -> 钉住前 %d 步" % (d, _RTC["extra"], _RTC["last_d"])

        if self.log_every and (self.n_req % self.log_every == 0):
            print("[rtc] #%d %s  (累计钉住 %d/%d 次)"
                  % (self.n_req, tag, _RTC["n_pinned"], _RTC["n_calls"]), flush=True)

        if self.jpeg:
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
    ap.add_argument("--no-jpeg", action="store_true")
    ap.add_argument("--no-rtc", action="store_true", help="关掉前缀约束（A/B 用，等价 100 号脚本）")
    ap.add_argument("--rtc-extra", type=int, default=0,
                    help="在客户端上报的已执行步数上再加几步；★ 前缀改成按 d 平移之后必须保持 0，否则钉错一位")
    ap.add_argument("--rtc-beta", type=float, default=1.0,
                    help="引导权重上限（接缝处）。越大越贴旧计划越平滑，越小越灵活")
    ap.add_argument("--rtc-no-decay", dest="rtc_decay", action="store_false", default=True,
                    help="关掉权重衰减（整个重叠区都用同一个 beta）")
    ap.add_argument("--rtc-log-every", type=int, default=1)
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

    _RTC["on"] = not a.no_rtc
    _RTC["extra"] = a.rtc_extra
    _RTC["beta"] = float(a.rtc_beta)
    _RTC["decay"] = bool(a.rtc_decay)
    if _RTC["on"]:
        install_rtc_patch(policy)

    wrapped = RTCJpegPolicy(policy, jpeg=not a.no_jpeg, log_every=a.rtc_log_every)
    print(
        "[server] rtc=%s(+%d)  jpeg=%s  model=%s  data_config=%s  denoise=%d  port=%d"
        % (_RTC["on"], a.rtc_extra, not a.no_jpeg, a.model_path, a.data_config,
           a.denoising_steps, a.port),
        flush=True,
    )
    RobotInferenceServer(wrapped, host=a.host, port=a.port, api_token=a.api_token).run()


if __name__ == "__main__":
    main()

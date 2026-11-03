#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GR00T 微调任务的实时监控面板（本机跑）。

本地起一个极小的 HTTP 服务，后台线程每 REFRESH 秒 SSH 到 AutoDL 执行
`/root/setup/status_gr00t.py` 拿一份 JSON 快照并缓存；页面轮询 /api/status 渲染。

★ 面板覆盖两段：**前置处理**（转换/规整/统一分辨率/裁静止段）与**训练**。

    <venv-python> scripts/dashboard_gr00t.py [port]        # 默认 8771

然后浏览器打开 http://localhost:<port>/
（连接信息统一读 scripts/conn.py）
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import paramiko

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from conn import HOST, PASSWORD, PORT, USER
except ImportError:
    HOST, PORT, USER, PASSWORD = "connect.bjb3.seetacloud.com", 30024, "root", "TJSz1x4wVs1Y"

REMOTE_STATUS = "/root/setup/status_gr00t.py"
REMOTE_PY = "/root/miniconda3/envs/gr00t/bin/python"

HTTP_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8771
REFRESH = 10

STATE = {"ok": False, "error": "initialising", "data": None, "fetched_at": 0}
LOCK = threading.Lock()


def fetch_once():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=25,
                    banner_timeout=60, auth_timeout=60, allow_agent=False, look_for_keys=False)
        env = {
            "GR00T_OUT": os.environ.get("GR00T_OUT", "/root/autodl-tmp/gr00t_trim2"),
            "GR00T_TRAIN_LOG": os.environ.get("GR00T_TRAIN_LOG", "/root/logs/train_trim2.log"),
            "GR00T_DATASET": os.environ.get("GR00T_DATASET", "/root/gr00t_data/arm101_pick_all_objects_trim2"),
            "GR00T_SCREEN": os.environ.get("GR00T_SCREEN", "trim2"),
            "GR00T_TOTAL_STEPS": os.environ.get("GR00T_TOTAL_STEPS", "30000"),
            "GR00T_EVAL_LOG": os.environ.get("GR00T_EVAL_LOG", "/root/logs/trim2_eval.log"),
        }
        cmd = " ".join("%s=%s" % (k, v) for k, v in env.items()) + \
              " %s %s 2>/dev/null" % (REMOTE_PY, REMOTE_STATUS)
        _in, out, err = cli.exec_command(cmd, timeout=60)
        raw = out.read().decode("utf-8", "replace").strip()
        if not raw:
            raise RuntimeError("empty output: %s" % err.read().decode("utf-8", "replace")[:200])
        data = json.loads(raw.splitlines()[-1])
        with LOCK:
            STATE.update(ok=True, error=None, data=data, fetched_at=time.time())
    except Exception as e:  # noqa: BLE001
        with LOCK:
            STATE.update(ok=False, error="%s: %s" % (type(e).__name__, e), fetched_at=time.time())
    finally:
        try:
            cli.close()
        except Exception:
            pass


def poller():
    while True:
        fetch_once()
        time.sleep(REFRESH)


PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GR00T N1.5 · 训练进度</title>
<style>
  :root{
    --bg:#f6f7f9; --card:#fff; --line:#e4e7ec; --text:#1b1f24; --muted:#667085;
    --red:#d92d20; --green:#0f9d58; --blue:#2563eb; --amber:#b54708; --ok:#067647;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
    font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
  .wrap{max-width:1120px;margin:0 auto;padding:22px 18px 60px}
  h1{font-size:20px;margin:0 0 4px;font-weight:650}
  .sub{color:var(--muted);font-size:12.5px;margin-bottom:18px}
  .grid{display:grid;gap:14px}
  .g2{grid-template-columns:repeat(2,minmax(0,1fr))}
  .g4{grid-template-columns:repeat(4,minmax(0,1fr))}
  @media(max-width:860px){.g2,.g4{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  .card h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
    margin:0 0 10px;font-weight:650}
  .kpi{font-size:24px;font-weight:700;letter-spacing:-.02em}
  .kpi small{font-size:13px;font-weight:500;color:var(--muted);letter-spacing:0}
  .pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:650}
  .p-run{background:#e8f1ff;color:#1d4ed8}
  .p-ok{background:#e7f7ee;color:#067647}
  .p-wait{background:#fef3e6;color:#b54708}
  .p-err{background:#fee4e2;color:#b42318}
  .bar{height:12px;background:#eceff3;border-radius:999px;overflow:hidden;margin:10px 0 6px}
  .bar > i{display:block;height:100%;background:linear-gradient(90deg,#2563eb,#4f8cff);width:0%;
    transition:width .5s ease}
  .bar > i.g{background:linear-gradient(90deg,#0f9d58,#34d399)}
  .bar > i.ind{width:100%!important;background:repeating-linear-gradient(90deg,#c7d7fe 0 12px,#eef4ff 12px 24px);
    animation:sl 1.1s linear infinite}
  @keyframes sl{from{background-position:0 0}to{background-position:24px 0}}
  .row{display:flex;justify-content:space-between;gap:10px;font-size:13px;padding:3px 0}
  .row span:first-child{color:var(--muted)}
  .row span:last-child{font-variant-numeric:tabular-nums;font-weight:600}
  pre{background:#0f1420;color:#d7e2f1;border-radius:10px;padding:12px;overflow:auto;
    max-height:260px;font-size:11.4px;line-height:1.5;margin:0;white-space:pre-wrap}
  .muted{color:var(--muted)}
  .mono{font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;vertical-align:1px}
  .ok{background:var(--ok)} .warn{background:#dc6803} .err{background:#d92d20} .idle{background:#98a2b3}
  .chip{font-size:12px;padding:5px 10px;border-radius:8px;background:#f2f4f7;color:#475467;border:1px solid var(--line)}
  .chip.done{background:#e7f7ee;color:#067647;border-color:#abefc6;font-weight:650}
  .chip.on{background:#e8f1ff;color:#1d4ed8;border-color:#c7d7fe;font-weight:650}
  .chip.bad{background:#fee4e2;color:#b42318;border-color:#fecdca;font-weight:650}
  .flow{display:flex;flex-wrap:wrap;gap:8px}
  svg.spark{display:block;width:100%;height:96px}
  .tag{font-size:11.5px;color:var(--muted)}
</style>
</head>
<body>
<div class="wrap">
  <h1>GR00T N1.5 · 官方 Isaac-GR00T 管道 · <span id="dsName">arm101_pick_all_objects</span></h1>
  <div class="sub">
    <span id="conn"><span class="dot idle"></span>连接中…</span>
    &nbsp;·&nbsp; 上次刷新 <span id="ts" class="mono">-</span>
    &nbsp;·&nbsp; 每 10s 自动刷新
    &nbsp;·&nbsp; <span class="mono" id="tb">-</span>
  </div>

  <div class="grid" style="margin-bottom:14px">
    <div class="card">
      <h2>整体阶段</h2>
      <div class="flow" id="flow"></div>
      <div style="margin-top:12px" id="phaseLine"><span class="pill p-wait">等待</span></div>
    </div>
  </div>

  <div class="card" style="margin-bottom:14px">
    <h2>探针评估 · 视觉消融（112_vision_ablation.py）</h2>
    <div class="row"><span>状态</span><span><span class="pill p-wait" id="peState">-</span></span></div>
    <table style="width:100%;border-collapse:collapse;font-size:12.5px;margin-top:8px">
      <thead><tr>
        <th style="text-align:left;font-weight:500;color:var(--muted);padding:4px 6px">采样阶段</th>
        <th style="text-align:right;font-weight:500;color:var(--muted);padding:4px 6px">跨样本变化</th>
        <th style="text-align:right;font-weight:500;color:var(--muted);padding:4px 6px">两路相机置黑引起</th>
        <th style="text-align:right;font-weight:500;color:var(--muted);padding:4px 6px">真图误差</th>
        <th style="text-align:right;font-weight:500;color:var(--muted);padding:4px 6px">置黑误差</th>
      </tr></thead>
      <tbody id="peBody"></tbody>
    </table>
    <div class="muted" style="font-size:12px;margin-top:8px" id="peHint">-</div>
    <pre id="peLog" style="max-height:150px;margin-top:10px">-</pre>
  </div>

  <div class="grid g4" style="margin-bottom:14px">
    <div class="card"><h2>训练步数</h2><div class="kpi" id="kStep">-<small></small></div>
      <div class="bar"><i id="pbar"></i></div>
      <div class="row"><span>进度</span><span id="kPct">-</span></div>
      <div class="row"><span>预计剩余</span><span id="kEta">-</span></div>
    </div>
    <div class="card"><h2>Loss（TensorBoard）</h2><div class="kpi" id="kLoss">-</div>
      <div class="row"><span>grad norm</span><span id="kGrad">-</span></div>
      <div class="row"><span>学习率</span><span id="kLr">-</span></div>
      <div class="row"><span>epoch</span><span id="kEpoch">-</span></div>
    </div>
    <div class="card"><h2 id="gpuTitle">GPU</h2><div class="kpi" id="kUtil">-<small> %</small></div>
      <div class="bar"><i id="gbar" class="g"></i></div>
      <div class="row"><span>显存</span><span id="kMem">-</span></div>
      <div class="row"><span>温度 / 功耗</span><span id="kTemp">-</span></div>
    </div>
    <div class="card"><h2>速度 / 产出</h2>
      <div class="row"><span>单步耗时</span><span id="kStepS">-</span></div>
      <div class="row"><span>已运行</span><span id="kElapsed">-</span></div>
      <div class="row"><span>输出体积</span><span id="kOut">-</span></div>
      <div class="row"><span>checkpoint</span><span id="kCkpt">-</span></div>
    </div>
  </div>

  <div class="grid g2" style="margin-bottom:14px">
    <div class="card"><h2>Loss 曲线</h2>
      <svg class="spark" id="spark" viewBox="0 0 600 96" preserveAspectRatio="none"></svg>
      <div class="row"><span id="sparkInfo" class="tag">-</span><span id="sparkLast" class="mono">-</span></div>
    </div>
    <div class="card"><h2>磁盘 / 数据集</h2><div id="disk"></div>
      <div id="ds" class="muted" style="margin-top:8px;font-size:12.5px"></div>
      <div id="ckptList" class="mono muted" style="margin-top:8px;font-size:12px"></div>
    </div>
  </div>

  <div class="grid g2" style="margin-bottom:14px">
    <div class="card"><h2>环境 / 组件版本</h2><div id="env"></div></div>
    <div class="card"><h2>正在运行的进程 / screen</h2>
      <div id="procs" class="muted mono" style="font-size:12px">-</div></div>
  </div>

  <div class="card"><h2 id="logTitle">训练日志</h2><pre id="log">-</pre></div>
</div>

<script>
const ORDER = ["prep","ready","training","done"];
const LABEL = {prep:"预处理", ready:"待起训", training:"训练", done:"完成"};
const PH2IDX = {prep:0, prep_failed:0, prep_done:1, idle:1, starting:2, training:2,
                stopped:2, error:2, done:3};
function el(id){return document.getElementById(id)}
function esc(s){return (s??"").toString().replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function fmt(n,d){ if(n==null)return "-"; return Number(n).toFixed(d==null?4:d) }
function fmtDur(s){ if(s==null) return "—"; s=Math.max(0,Math.round(s));
  const h=Math.floor(s/3600), m=Math.floor((s%3600)/60), ss=s%60;
  return h? (h+"h "+m+"m") : (m? (m+"m "+ss+"s") : (ss+"s")); }

const PE_BASE = {   // v3（冻结视觉）的基线，用于对比"解冻后是否真的开始看图"
  early: {cross:56.6, black_both:2.19, err_true:1.34, err_black:2.14},
  mid:   {cross:33.1, black_both:2.66, err_true:1.16, err_black:1.56}
};

function renderProbeEval(d){
  const pe = d.probe_eval||{}, st = el("peState");
  if(!pe.exists){ st.className="pill p-wait"; st.textContent="等训练结束后自动开始"; }
  else if(pe.failed){ st.className="pill p-err"; st.textContent="评估失败（见下方日志）"; }
  else if(pe.done){ st.className="pill p-ok"; st.textContent="已完成"; }
  else { st.className="pill p-run"; st.textContent="等训练结束 / 评估中…"; }

  const rows = (pe.phases||[]).map(p=>({phase:p.phase, cross:p.cross, black_both:p.black_both,
                                        err_true:p.err_true, err_black:p.err_black, base:PE_BASE[p.phase]}));
  if(!rows.length){
    rows.push({phase:"early（v3 基线）", ...PE_BASE.early});
    rows.push({phase:"mid（v3 基线）",   ...PE_BASE.mid});
  }
  el("peBody").innerHTML = rows.map(r=>{
    const dl = (r.black_both!=null && r.base && r.base.black_both!=null) ? r.black_both - r.base.black_both : null;
    const bg = (dl!=null && dl>3) ? ' style="background:#ecfdf3"' : "";
    return `<tr${bg}>
      <td style="padding:4px 6px">${esc(r.phase)}</td>
      <td class="mono" style="padding:4px 6px;text-align:right">${fmt(r.cross,1)}</td>
      <td class="mono" style="padding:4px 6px;text-align:right"><b>${fmt(r.black_both,2)}</b>` +
      (dl!=null? ` <span class="muted">(${dl>0?"+":""}${fmt(dl,2)})</span>`:"") + `</td>
      <td class="mono" style="padding:4px 6px;text-align:right">${fmt(r.err_true,2)}</td>
      <td class="mono" style="padding:4px 6px;text-align:right">${fmt(r.err_black,2)}</td>
    </tr>`;
  }).join("");
  el("peHint").textContent =
    "判据：v3 基线把两路相机全置黑只改 2.19°（early）—— 探针若把它推到 5~10°+ ⇒ 解冻视觉真的开始看图，值得跑完整 30k；" +
    "若仍是 1~3° ⇒ 问题在数据/相机，重训无用。";
  el("peLog").textContent = (pe.tail||[]).join("\n") || "-";
}

function drawSpark(series){
  const svg = el("spark"); svg.innerHTML = "";
  if(!series || series.length<2){ el("sparkInfo").textContent="等待数据（每 10 步记录一次）"; el("sparkLast").textContent="-"; return; }
  const W=600,H=96,pad=6;
  const xs=series.map(p=>p[0]), ys=series.map(p=>p[1]);
  const x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys), y1=Math.max(...ys);
  const sx=v=>pad+(W-2*pad)*(x1===x0?0:(v-x0)/(x1-x0));
  const sy=v=>H-pad-(H-2*pad)*(y1===y0?0.5:(v-y0)/(y1-y0));
  const d=series.map((p,i)=>(i?"L":"M")+sx(p[0]).toFixed(1)+" "+sy(p[1]).toFixed(1)).join(" ");
  const ns="http://www.w3.org/2000/svg";
  const area=document.createElementNS(ns,"path");
  area.setAttribute("d", d+` L ${sx(x1).toFixed(1)} ${H-pad} L ${sx(x0).toFixed(1)} ${H-pad} Z`);
  area.setAttribute("fill","rgba(37,99,235,.10)"); svg.appendChild(area);
  const line=document.createElementNS(ns,"path");
  line.setAttribute("d", d); line.setAttribute("fill","none");
  line.setAttribute("stroke","#2563eb"); line.setAttribute("stroke-width","2");
  line.setAttribute("stroke-linejoin","round"); svg.appendChild(line);
  for(const g of [0.25,0.5,0.75]){
    const l=document.createElementNS(ns,"line");
    l.setAttribute("x1",pad); l.setAttribute("x2",W-pad);
    l.setAttribute("y1",pad+(H-2*pad)*g); l.setAttribute("y2",pad+(H-2*pad)*g);
    l.setAttribute("stroke","#eef1f5"); l.setAttribute("stroke-width","1"); svg.insertBefore(l,svg.firstChild);
  }
  const last=series[series.length-1];
  el("sparkInfo").textContent = `step ${xs[0]} → ${last[0]} ｜ 区间 ${y0.toFixed(4)} ~ ${y1.toFixed(4)}（越低越好）`;
  el("sparkLast").textContent = `最新 loss ${last[1].toFixed(4)}`;
}

async function tick(){
  try{
    const r = await fetch("/api/status",{cache:"no-store"});
    const j = await r.json();
    if(!j.ok){ el("conn").innerHTML='<span class="dot err"></span>'+esc(j.error||"unknown"); return; }
    const d = j.data;
    el("conn").innerHTML='<span class="dot ok"></span>已连接';
    el("ts").textContent = d.ts_str;
    el("tb").textContent = d.tb_error ? ("TB 读取失败: "+d.tb_error) : ((d.tb_file||"").split("/").slice(-2).join("/")||"-");
    if(d.dataset_dir) el("dsName").textContent = d.dataset_dir.split("/").slice(-1)[0];

    // ---- 整体阶段
    const idx = PH2IDX[d.phase] != null ? PH2IDX[d.phase] : 1;
    el("flow").innerHTML = ORDER.map((p,i)=>{
      const bad = (d.phase==="prep_failed" && i===0);
      let cls = bad?"chip bad":(i<idx?"chip done":(i===idx?"chip on":"chip"));
      return `<span class="${cls}">${LABEL[p]}</span>`;
    }).join("");
    let pcls="p-wait", ptxt="等待";
    if(d.phase==="training"){pcls="p-run";ptxt="训练中（screen "+esc(d.screen||"")+"）"}
    else if(d.phase==="starting"){pcls="p-run";ptxt="启动中（加载模型…）"}
    else if(d.phase==="done"){pcls="p-ok";ptxt="训练完成"}
    else if(d.phase==="error"){pcls="p-err";ptxt="出错（日志有 Traceback，见下）"}
    else if(d.phase==="stopped"){pcls="p-wait";ptxt="训练进程已退出"}
    else if(d.phase==="prep_failed"){pcls="p-err";ptxt="预处理失败（PREP_FAILED，见预处理日志）"}
    else if(d.phase==="prep_done"){pcls="p-ok";ptxt="预处理完成，等待自动接力起训…"}
    else if(d.phase==="prep"){pcls="p-run";ptxt="预处理中"}
    el("phaseLine").innerHTML = `<span class="pill ${pcls}">${ptxt}</span>`;

    // ---- 探针评估（视觉消融）
    renderProbeEval(d);

    // ---- 步数
    const t = d.train||{}, total = d.total_steps||30000;
    const cur = t.step;
    el("kStep").innerHTML = (cur==null?"-":cur) + `<small> / ${total}</small>`;
    const pct = (t.pct!=null? t.pct : (cur!=null? cur/total*100 : 0));
    el("pbar").style.width = Math.min(100,pct).toFixed(1)+"%";
    el("kPct").textContent = pct.toFixed(1)+"%";
    el("kEta").textContent = t.eta_h || "-";

    el("kLoss").textContent = fmt(t.loss);
    el("kGrad").textContent = fmt(t.grad_norm,3);
    el("kLr").textContent = t.lr ?? "-";
    el("kEpoch").textContent = fmt(t.epoch,3);
    el("kStepS").textContent = t.step_s ? t.step_s+" s/step" : "-";
    el("kElapsed").textContent = (d.bar&&d.bar.elapsed) ? d.bar.elapsed : "-";
    el("kOut").textContent = d.out_size || "-";
    const cks = d.checkpoints||[];
    el("kCkpt").textContent = cks.length ? cks.map(c=>c.step).join(", ") : "尚无";
    el("ckptList").innerHTML = cks.length
      ? cks.map(c=>`checkpoint-${c.step}  ${c.size||""}`).join("<br>")
      : (d.final_model? "已有最终模型（output_dir 根）" : "尚无 checkpoint（每 2000 步存一次）");

    // ---- GPU
    const g = d.gpu||{};
    el("gpuTitle").textContent = "GPU · " + (g.name||"-");
    el("kUtil").innerHTML = (g.util!=null? g.util : "-") + `<small> %</small>`;
    el("gbar").style.width = (g.util||0)+"%";
    el("kMem").textContent = (g.mem_used!=null? g.mem_used+" / "+g.mem_total+" MiB ("+g.mem_pct+"%)" : "-");
    el("kTemp").textContent = (g.temp!=null? g.temp+"°C / "+Number(g.power||0).toFixed(0)+"W" : "-");

    drawSpark(d.series);

    el("disk").innerHTML = (d.disk||[]).map(l=>`<div class="row"><span class="mono" style="font-size:11.5px">${esc(l)}</span></div>`).join("");
    el("ds").textContent = "数据集: " + (d.dataset_ok? "✓ " + d.dataset_dir + "  " + (d.dataset_size||"") : "✗ 缺失（预处理尚未产出）");

    const e=d.env_pkgs||{};
    el("env").innerHTML = Object.keys(e).sort().map(k=>`<div class="row"><span>${esc(k)}</span><span class="mono">${esc(e[k])}</span></div>`).join("") || '<span class="muted">-</span>';

    el("procs").innerHTML = (d.screens||[]).map(s=>esc(s)).join("<br>") +
      ((d.procs||[]).length? "<br>"+(d.procs||[]).map(p=>esc(p.slice(0,110))).join("<br>") : "");

    el("logTitle").textContent = "训练日志 · " + (d.train_log||"");
    el("log").textContent = (d.train_tail||[]).join("\n") || "-";
  }catch(err){
    el("conn").innerHTML='<span class="dot err"></span>'+esc(err);
  }
}
tick(); setInterval(tick, 10000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/status"):
            with LOCK:
                payload = dict(STATE)
                payload["age"] = round(time.time() - payload.get("fetched_at", 0), 1)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def main():
    threading.Thread(target=poller, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), Handler)
    print("dashboard: http://localhost:%d/  (remote %s:%s)" % (HTTP_PORT, HOST, PORT), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

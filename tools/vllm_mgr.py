#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vllm_mgr.py - supervisor for the vllm_kestrel engine (RK3588 board).

A tiny zero-dependency (stdlib-only) process manager that:
  * starts / stops / restarts the engine from the persisted config file
    written by the engine's /admin/api/config/save endpoint
  * runs an NPU calibration self-test (--npu-calib) in a subprocess
  * serves a minimal HTTP status page (default port 8082) so the admin
    page at :8080/admin/ can start the engine after a shutdown

Usage:
  python3 vllm_mgr.py status                 # show engine status
  python3 vllm_mgr.py start                  # launch the engine (nohup)
  python3 vllm_mgr.py stop                   # stop it (SIGTERM -> SIGKILL)
  python3 vllm_mgr.py restart                # stop then start
  python3 vllm_mgr.py selftest               # run --npu-calib, print result
  python3 vllm_mgr.py serve [--serve 8082]   # HTTP supervisor (daemon loop)

Config file schema (written by the admin page):
  { "model_dir": "<path-to-model-dir>", "wmode": "q4", "kv_q4": false,
    "prefix_kv": true, "prefix_cache": false, "prefill_q8": false,
    "sparse_attn": false,
    "sparse_k": 32, "npu": true, "npu_load": 1, "npu_infer": 1,
    "npu_timing": 0, "port": 8080, "threads": 8, "max_queued": 16,
    "min_free_mb": 2048, "env": {"OMP_NUM_THREADS": "4"} }
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
# Repo root (one level up from tools/), overridable for board layouts that do
# not match the repository layout (e.g. a dedicated deploy directory).
REPO = os.environ.get("VLLM_MGR_REPO", os.path.dirname(HERE))

def env_path(name, dflt):
    v = os.environ.get(name)
    return v if v else dflt

def config_path():
    return env_path("VLLM_ADMIN_CONFIG", os.path.join(REPO, "webmgr", "config.json"))

def log_path():
    return env_path("VLLM_ADMIN_LOG", os.path.join(REPO, "logs", "engine.log"))

def flag_path():
    return env_path("VLLM_MGR_FLAG", os.path.join(REPO, "webmgr", "restart.flag"))

def find_bin():
    v = os.environ.get("VLLM_MGR_BIN")
    if v and os.path.isfile(v):
        return v
    for c in (os.path.join(HERE, "..", "build-rk3588", "vllm_kestrel"),
              os.path.join(HERE, "..", "build", "vllm_kestrel"),
              os.path.join(REPO, "build-rk3588", "vllm_kestrel"),
              "./build-rk3588/vllm_kestrel"):
        if os.path.isfile(c):
            return c
    return os.path.join(REPO, "build-rk3588", "vllm_kestrel")


def load_config():
    p = config_path()
    if not os.path.isfile(p):
        return None, p
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f), p
    except Exception:
        return None, p


def build_cmd(cfg, bin_):
    cmd = [bin_, "--serve",
           "--port", str(int(cfg.get("port", 8080))),
           "--model", cfg.get("model_dir", os.path.join(REPO, "models", "qwen3-vl-2b")),
           "--wmode", cfg.get("wmode", "q4"),
           "--threads", str(int(cfg.get("threads", 8))),
           "--max-queued", str(int(cfg.get("max_queued", 16))),
           "--min-free-mb", str(int(cfg.get("min_free_mb", 2048)))]
    if cfg.get("kv_q4"):
        cmd.append("--kv-q4")
    # KV 前缀复用默认开（引擎默认）；管理页取消勾选（prefix_kv=false）时才关闭
    if cfg.get("prefix_kv", True) is False:
        cmd.append("--no-prefix-kv")
    if cfg.get("prefix_cache"):
        cmd.append("--prefix-cache")
    if cfg.get("prefill_q8"):
        cmd.append("--prefill-q8")
    if cfg.get("sparse_attn"):
        cmd += ["--sparse-attn", "--sparse-k", str(int(cfg.get("sparse_k", 32)))]
    if cfg.get("l3_evict"):
        if not cfg.get("sparse_attn"):
            # L3 cold-block eviction needs sparse decode attention
            cmd += ["--sparse-attn"]
        cmd += ["--l3-evict"]
        cmd += ["--l3-ratio", str(float(cfg.get("l3_ratio", 0.75)))]
        cmd += ["--l3-min-seq", str(int(cfg.get("l3_min_seq", 0)))]
        p = str(cfg.get("l3_path") or "").strip()
        if p:
            cmd += ["--l3-path", p]   # L3 cache file path (e.g. eMMC mount)
        ms = int(cfg.get("l3_max_size") or 0)
        if ms > 0:
            cmd += ["--l3-max-size", str(ms)]   # size cap in MB (0 = unlimited)
        ttl = int(cfg.get("l3_user_ttl") or 30)
        if ttl > 0:
            cmd += ["--l3-user-ttl", str(ttl)]  # per-user file idle TTL in minutes
    if cfg.get("npu"):
        cmd.append("--npu")
    if cfg.get("device"):
        cmd += ["--device", str(cfg["device"])]   # device profile override
    if cfg.get("disk_kv"):
        # Disk KV persistence: F32 conversation snapshots survive restarts.
        d = str(cfg.get("disk_kv_dir") or "").strip()
        cmd += ["--disk-kv", d or "kv_disk"]
    if cfg.get("spec"):
        # n-gram speculative decode (greedy-only) with draft length --spec-k.
        cmd += ["--spec", "--spec-k", str(int(cfg.get("spec_k", 4)))]
    mp = cfg.get("min_p")
    if mp:
        cmd += ["--min-p", str(float(mp))]
    # Load-on-use / unload-when-idle (用时加载 / 不用时卸载): the model is
    # loaded on the first request and auto-unloaded after idle seconds.
    if cfg.get("auto_load"):
        cmd.append("--load-on-use")
    au = int(cfg.get("auto_unload_s") or 0)
    if au > 0:
        cmd += ["--auto-unload", str(au)]
    # Model load format (auto / vqf). 引擎已为纯 VQF 运行时。
    lf = str(cfg.get("format") or "auto").strip().lower()
    if lf in ("vqf",):
        cmd += ["--load-format", lf]
    if not cfg.get("auto_start"):
        # Manual mode: the engine starts with the model NOT loaded; the
        # admin page triggers the load (with a chosen quantization mode).
        cmd.append("--no-auto-load")
    return cmd


def build_env(cfg):
    env = dict(os.environ)
    env["VLLM_NPU_LOAD"] = str(int(cfg.get("npu_load", 0)))
    env["VLLM_NPU_INFER"] = str(int(cfg.get("npu_infer", 1)))
    env["VLLM_NPU_TIMING"] = str(int(cfg.get("npu_timing", 0)))
    env["VLLM_NPU_WCACHE"] = str(int(cfg.get("npu_wcache", 0)))
    e = cfg.get("env") or {}
    for k, v in e.items():
        env[str(k)] = str(v)
    env.setdefault("OMP_NUM_THREADS", "4")
    return env


def engine_pids():
    """PIDs of the running engine (pgrep on the --serve command line)."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "vllm_kestrel --serve"],
            stderr=subprocess.DEVNULL, text=True)
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:
        return []


def start_engine():
    cfg, path = load_config()
    if cfg is None:
        return {"ok": False,
                "error": "no config at %s - save it from the admin page first" % path}
    if engine_pids():
        return {"ok": False, "error": "engine already running"}
    bin_ = find_bin()
    if not os.path.isfile(bin_):
        return {"ok": False, "error": "engine binary not found: %s" % bin_}
    cmd = build_cmd(cfg, bin_)
    env = build_env(cfg)
    os.makedirs(os.path.dirname(log_path()), exist_ok=True)
    try:
        with open(log_path(), "ab") as logf:
            logf.write(b"\n===== engine start %s =====\n" %
                       time.strftime("%F %T").encode())
            logf.flush()
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
    except Exception as e:
        return {"ok": False, "error": "launch failed: %s" % e}
    return {"ok": True, "pid": p.pid, "cmd": " ".join(cmd)}


def stop_engine(timeout=12):
    pids = engine_pids()
    if not pids:
        return {"ok": True, "note": "engine not running"}
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    for _ in range(timeout * 2):
        if not engine_pids():
            return {"ok": True, "stopped": pids}
        time.sleep(0.5)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    return {"ok": True, "stopped": pids, "note": "SIGKILL after SIGTERM timeout"}


def request_restart():
    f = flag_path()
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, "w") as fh:
        fh.write(time.strftime("%F %T"))
    return {"ok": True, "note": "restart armed: engine will relaunch after shutdown"}


def run_selftest(timeout=240):
    bin_ = find_bin()
    cmd = [bin_, "--npu", "--npu-calib"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        tail = [ln for ln in out.strip().splitlines() if ln.strip()][-6:]
        summary = " | ".join(tail)
        ok = (p.returncode == 0 and "[NPU CALIBRATION COMPLETE]" in out)
        return {"ok": ok, "result": summary, "rc": p.returncode}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "selftest timed out (%ss)" % timeout}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def status():
    pids = engine_pids()
    cfg, cpath = load_config()
    return {
        "engine_running": bool(pids),
        "pids": pids,
        "config": cpath,
        "config_exists": cfg is not None,
        "binary": find_bin(),
        "restart_armed": os.path.isfile(flag_path()),
        "time": time.strftime("%F %T"),
    }


# ---------------- NPU driver check / install ----------------

DRM_NODES = ("/dev/dri/renderD128", "/dev/dri/renderD129")
MODPROBE_CANDIDATES = ("rknpu_drm", "rknpu", "rockchipnpu", "rknpu_rockchip")


def _drm_driver(node="/dev/dri/renderD129"):
    """driver name from the DRM device's sysfs chain. /dev/dri/renderD129 is a
    char device, so resolve via /sys/class/drm/renderD129:
    -> /sys/devices/platform/fdab0000.npu/drm/renderD129
    -> device/driver symlink basename (e.g. RKNPU); driver/uevent may be
    unreadable (sysfs permission) so the symlink is the primary source."""
    try:
        cl = "/sys/class/drm/" + os.path.basename(node)
        real = os.path.realpath(cl)
        dev = os.path.dirname(os.path.dirname(real))
        d = os.path.join(dev, "driver")
        if os.path.islink(d):
            return os.path.basename(os.path.realpath(d))
        ue = os.path.join(dev, "driver", "uevent")
        if os.path.isfile(ue):
            with open(ue, "r") as f:
                for ln in f:
                    if ln.startswith("DRIVER="):
                        return ln.split("=", 1)[1].strip()
    except Exception:
        pass
    return None


def npu_drv_info():
    checks = []

    def chk(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    try:
        chk("内核版本", True,
            subprocess.check_output(["uname", "-r"], text=True).strip())
    except Exception as e:
        chk("内核版本", False, e)

    for node in DRM_NODES:
        try:
            st = os.stat(node)
            chk(os.path.basename(node), True,
                "mode=%s grp=%s" % (oct(st.st_mode & 0o777), st.st_gid))
        except OSError as e:
            chk(os.path.basename(node), False, e.strerror or str(e))

    chk("/dev/rknpu", os.path.exists("/dev/rknpu"),
        "存在" if os.path.exists("/dev/rknpu") else "不存在 (DRM 后端模式)")

    drv = _drm_driver()
    chk("DRM 驱动", drv is not None, drv or "无法读取 sysfs driver")

    dt = []
    try:
        dt = [x for x in os.listdir("/proc/device-tree")
              if "npu" in x.lower()]
    except Exception:
        pass
    chk("设备树 NPU 节点", bool(dt),
        "; ".join(dt) or "未找到 (需内核 dts 支持)")

    open_ok = False
    open_err = ""
    try:
        fd = os.open("/dev/dri/renderD129", os.O_RDWR)
        os.close(fd)
        open_ok = True
    except OSError as e:
        open_err = e.strerror or str(e)
    chk("renderD129 打开测试", open_ok, "OK" if open_ok else open_err)

    chk("librknnrt 用户态库", os.path.exists("/usr/lib/librknnrt.so"),
        "存在" if os.path.exists("/usr/lib/librknnrt.so") else "不存在 (direct 后端不依赖)")

    dmesg_lines = []
    try:
        out = subprocess.check_output(["dmesg"], text=True,
                                      stderr=subprocess.DEVNULL)
        dmesg_lines = [ln for ln in out.splitlines()
                       if re.search(r"rknpu|npu", ln, re.I)][-3:]
    except Exception:
        pass
    chk("dmesg NPU 日志", bool(dmesg_lines),
        " | ".join(dmesg_lines) if dmesg_lines else "无相关日志")

    overall = "ok" if open_ok and drv else ("warn" if open_ok else "error")
    return {"ok": True, "overall": overall, "checks": checks}


def npu_drv_install():
    steps = []
    for mod in MODPROBE_CANDIDATES:
        rc = subprocess.call(["modprobe", mod], stderr=subprocess.DEVNULL)
        if rc == 0:
            steps.append({"step": "modprobe %s" % mod, "ok": True,
                          "detail": "加载成功"})
            break
        steps.append({"step": "modprobe %s" % mod, "ok": False,
                      "detail": "模块不存在或已内置 (rc=%d)" % rc})
    for node in DRM_NODES:
        if os.path.exists(node):
            try:
                os.chmod(node, 0o666)
                steps.append({"step": "chmod %s" % node, "ok": True,
                              "detail": "mode=666"})
            except OSError as e:
                steps.append({"step": "chmod %s" % node, "ok": False,
                              "detail": str(e)})
    info = npu_drv_info()
    last_open = [c for c in info["checks"]
                 if c["name"] == "renderD129 打开测试"]
    open_ok = last_open[0]["ok"] if last_open else False
    return {
        "ok": True,
        "open_test": open_ok,
        "overall": info["overall"],
        "steps": steps,
        "note": ("驱动就绪" if open_ok else
                 "仍需内核驱动：请确认内核包含 rockchip NPU 驱动"
                 "（rockchip 6.1/6.11 内核 + dts 节点），固件无法自动安装。"),
    }


# ---------------- HTTP supervisor ----------------

def page_html():
    st = status()
    color = "#3ecf8e" if st["engine_running"] else "#ffb020"
    state = "运行中" if st["engine_running"] else "已停止"
    eng = ("<a href='http://%s:8080/admin/' target='_blank'>打开完整管理页 (:8080/admin/)</a>"
           % (os.environ.get("HOSTNAME", "localhost")))
    if not st["engine_running"] and not st["config_exists"]:
        eng += "<p style='color:#888'>尚未保存配置 - 先在引擎运行期间打开管理页保存配置。</p>"
    return """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>vLLM-Kestrel supervisor</title><style>
body{background:#0f1419;color:#d7e0ea;font-family:Consolas,"Microsoft YaHei",monospace;padding:30px;max-width:640px;margin:0 auto}
h1{font-size:18px}.st{font-size:16px;margin:14px 0}.dot{display:inline-block;width:10px;height:10px;border-radius:50%%;margin-right:8px;background:%s}
button{background:#202a36;color:#d7e0ea;border:1px solid #2c3a4a;border-radius:6px;padding:8px 16px;margin:4px;font-size:14px;cursor:pointer}
button:hover{border-color:#4da3ff}.k{color:#8ea0b4;font-size:12px}</style></head><body>
<h1>vLLM-Kestrel supervisor</h1>
<div class="st"><span class="dot"></span>引擎状态: <b>%s</b> <span id="pid">%s</span></div>
<div class="k">%s</div>
<div style="margin-top:16px">
<button onclick="post('/api/start')">启动引擎</button>
<button onclick="post('/api/stop')">停止引擎</button>
<button onclick="post('/api/restart')">重启</button>
<button onclick="post('/api/selftest')">NPU 自测</button>
</div>
<div id="out" style="margin-top:12px;font-size:13px;white-space:pre-wrap"></div>
<script>
async function post(p){var o=document.getElementById('out');o.textContent='...';
try{var r=await fetch(p,{method:'POST'});var j=await r.json();
o.textContent=JSON.stringify(j,null,2);}catch(e){o.textContent='ERR '+e;}
setTimeout(()=>location.reload(),1200);}
</script></body></html>""" % (color, state, st["pids"], eng)


class MgrHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[mgr] %s\n" % (fmt % args))

    def _cors(self):
        # allow the engine admin page (:8080) to call this supervisor (:8082)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/":
            self._html(page_html())
        elif p.path == "/api/status":
            self._json(status())
        elif p.path == "/api/npu-info":
            self._json(npu_drv_info())
        else:
            self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        p = urlparse(self.path)
        if p.path == "/api/start":
            res = start_engine()
        elif p.path == "/api/stop":
            res = stop_engine()
        elif p.path == "/api/restart":
            res = request_restart()
        elif p.path == "/api/selftest":
            res = run_selftest()
        elif p.path == "/api/npu-install":
            res = npu_drv_install()
        else:
            res = {"ok": False, "error": "not found"}
        self._json(res)


def supervise_loop(stop_event):
    """Supervision loop:
      * auto_relaunch the engine when restart.flag is present and the engine
        is down (the admin page arms it before shutting the engine down);
      * if config.auto_start is enabled, automatically start the engine when
        it is down (with the model loaded: no --no-auto-load is passed)."""
    while not stop_event.wait(2.0):
        try:
            if engine_pids():
                continue
            if os.path.isfile(flag_path()):
                os.remove(flag_path())
                res = start_engine()
                print("[mgr] auto-restart: %s" % json.dumps(res, ensure_ascii=False),
                      flush=True)
                continue
            cfg, _ = load_config()
            if cfg and cfg.get("auto_start"):
                res = start_engine()
                print("[mgr] auto_start: %s" % json.dumps(res, ensure_ascii=False),
                      flush=True)
        except Exception as e:
            print("[mgr] supervise loop error: %s" % e, flush=True)


def serve_http(port):
    srv = ThreadingHTTPServer(("0.0.0.0", port), MgrHandler)
    stop_event = threading.Event()
    t = threading.Thread(target=supervise_loop, args=(stop_event,), daemon=True)
    t.start()
    print("[mgr] supervisor listening on :%d  (admin page: engine :8080/admin/)" % port,
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()


def main():
    ap = argparse.ArgumentParser(description="vllm_kestrel supervisor")
    ap.add_argument("action", nargs="?", default="status",
                    choices=["start", "stop", "restart", "status", "selftest", "serve"])
    ap.add_argument("--serve", nargs="?", const=8082, type=int, default=None,
                    help="run the HTTP supervisor (default port 8082)")
    args = ap.parse_args()

    if args.serve is not None:
        serve_http(args.serve)
        return

    if args.action == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    elif args.action == "start":
        print(json.dumps(start_engine(), ensure_ascii=False, indent=2))
    elif args.action == "stop":
        print(json.dumps(stop_engine(), ensure_ascii=False, indent=2))
    elif args.action == "restart":
        print(json.dumps(stop_engine(), ensure_ascii=False, indent=2))
        print(json.dumps(start_engine(), ensure_ascii=False, indent=2))
    elif args.action == "selftest":
        print(json.dumps(run_selftest(), ensure_ascii=False, indent=2))
    elif args.action == "serve":
        serve_http(8082)


if __name__ == "__main__":
    main()

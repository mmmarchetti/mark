"""In-process metrics for the web panel + live dashboard.

Tracks: uptime, turn count, per-stage pipeline latency (STT/LLM/TTS/TURN), tool
usage, per-backend LLM stats (local vs cloud: latency, tokens, fallbacks) with a
token-based cost estimate, local-server health, and a background SystemSampler
that polls GPU (nvidia-smi), CPU (psutil, per-core) and RAM into ring buffers.

psutil is expected on the Dell; GPU uses nvidia-smi. Everything degrades to None
if a source is unavailable - the dashboard just hides that panel.
"""

import collections
import logging
import os
import subprocess
import threading
import time

from reachy_mini_brain import config

try:
    import psutil
except Exception:  # pragma: no cover - psutil is expected but never hard-fail
    psutil = None

logger = logging.getLogger(__name__)

_HIST = 90  # ring-buffer length for the sparklines (~3 min at 2s cadence)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(len(s) * 0.95))], 2)


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


class _Backend:
    """Rolling stats for one LLM backend (local or cloud)."""

    def __init__(self) -> None:
        self.calls = 0
        self.latencies: collections.deque[float] = collections.deque(maxlen=50)
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def as_dict(self, price_in=0.0, price_out=0.0) -> dict:
        lat = list(self.latencies)
        d = {
            "calls": self.calls,
            "avg_latency_s": _avg(lat),
            "p95_latency_s": _p95(lat),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
        if price_in or price_out:
            d["est_cost_usd"] = round(
                self.prompt_tokens / 1e6 * price_in
                + self.completion_tokens / 1e6 * price_out, 4)
        return d


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start = time.time()
        self.turns = 0
        self._latencies: collections.deque[float] = collections.deque(maxlen=50)
        self.tool_counts: dict[str, int] = collections.defaultdict(int)
        # Per-stage pipeline latency (STT / LLM / TTS / TURN), rolling.
        self._stages: dict[str, collections.deque[float]] = collections.defaultdict(
            lambda: collections.deque(maxlen=50))
        # Per-backend LLM metrics.
        self._backends: dict[str, _Backend] = {"local": _Backend(), "cloud": _Backend()}
        self._fallbacks = 0
        # Local server health, set by the keep-warm pinger in main.py.
        self._local_health = {"reachable": None, "model": None, "checked": 0.0}
        # M4 (LLM host) system metrics, pulled by the poller in main.py from the
        # M4's read-only metrics agent. reachable=None until the first poll.
        self._m4 = {"reachable": None, "host": None, "cpu": None, "ram": None,
                    "gpu": None, "llm_proc": None, "checked": 0.0}
        self._m4_gpu_hist: collections.deque = collections.deque(maxlen=_HIST)
        # M4 RAM% and the mlx_lm.server process RSS (GB) over time - the latter is
        # the KV-cache-growth / OOM signal we added to the M4 agent.
        self._m4_ram_hist: collections.deque = collections.deque(maxlen=_HIST)
        self._m4_llm_rss_hist: collections.deque = collections.deque(maxlen=_HIST)
        # System sampler ring buffers (filled by SystemSampler).
        self._sys_lock = threading.Lock()
        self._gpu_hist: collections.deque = collections.deque(maxlen=_HIST)
        self._cpu_hist: collections.deque = collections.deque(maxlen=_HIST)
        self._ram_hist: collections.deque = collections.deque(maxlen=_HIST)
        self._gpu_latest: dict | None = None
        self._cpu_latest: dict | None = None
        self._ram_latest: dict | None = None
        self._sampler: SystemSampler | None = None

    # ---- recording (called from the brain loop / router / pingers) ----------

    def record_turn(self, latency_s: float, tools: list[str] | None = None) -> None:
        with self._lock:
            self.turns += 1
            self._latencies.append(latency_s)
            self._stages["turn"].append(latency_s)
            for t in (tools or []):
                self.tool_counts[t] += 1

    def record_stage(self, stage: str, latency_s: float) -> None:
        """Record a pipeline stage latency, e.g. 'stt', 'tts'."""
        with self._lock:
            self._stages[stage].append(latency_s)

    def record_llm_call(self, backend: str, latency_s: float,
                        prompt_tokens=None, completion_tokens=None) -> None:
        with self._lock:
            b = self._backends.get(backend)
            if b is None:
                b = self._backends[backend] = _Backend()
            b.calls += 1
            b.latencies.append(latency_s)
            if prompt_tokens:
                b.prompt_tokens += int(prompt_tokens)
            if completion_tokens:
                b.completion_tokens += int(completion_tokens)
            self._stages["llm"].append(latency_s)

    def record_llm_fallback(self) -> None:
        with self._lock:
            self._fallbacks += 1

    def record_local_health(self, reachable: bool, model: str | None = None) -> None:
        with self._lock:
            self._local_health = {"reachable": reachable, "model": model,
                                  "checked": time.time()}

    def record_m4(self, reachable: bool, data: dict | None = None) -> None:
        """Store the latest M4 (LLM host) CPU/RAM/GPU sample from its agent.

        `data` is the agent's /metrics JSON (host/cpu/ram/gpu); None when the
        poll failed (agent down / M4 asleep) -> reachable=False, panel greys out.
        """
        data = data or {}
        gpu = data.get("gpu") or None
        ram = data.get("ram") or None
        # The M4 agent's mlx-process memory field: accept either name so the
        # dashboard works regardless of which agent build is deployed there
        # (`llm_proc` from the staged version, `llm` from the M4's own build).
        llm_proc = data.get("llm_proc") or data.get("llm") or None
        with self._sys_lock:
            self._m4 = {
                "reachable": bool(reachable),
                "host": data.get("host"),
                "cpu": data.get("cpu"),
                "ram": ram,
                "gpu": gpu,
                "llm_proc": llm_proc,
                "checked": time.time(),
            }
            if reachable and gpu and gpu.get("util") is not None:
                self._m4_gpu_hist.append(gpu.get("util"))
            if reachable and ram and ram.get("percent") is not None:
                self._m4_ram_hist.append(ram.get("percent"))
            # Track the mlx_lm.server RSS so the growth is visible; append 0 when
            # the server is down so a gap/drop in the trend reads as "restarted".
            if reachable:
                self._m4_llm_rss_hist.append(
                    llm_proc.get("rss_gb") if llm_proc else 0)

    def push_system_sample(self, gpu: dict | None, cpu: dict | None, ram: dict | None) -> None:
        """Called by SystemSampler with the latest GPU/CPU/RAM readings."""
        with self._sys_lock:
            if gpu is not None:
                self._gpu_latest = gpu
                self._gpu_hist.append(gpu.get("util"))
            if cpu is not None:
                self._cpu_latest = cpu
                self._cpu_hist.append(cpu.get("overall"))
            if ram is not None:
                self._ram_latest = ram
                self._ram_hist.append(ram.get("percent"))

    def start_system_sampler(self, interval_s: float = 2.0, stop_event=None):
        self._sampler = SystemSampler(self, interval_s, stop_event)
        self._sampler.start()
        return self._sampler

    # ---- snapshot (served at /api/metrics) ----------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            up = int(time.time() - self._start)
            top = sorted(self.tool_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
            stages = {}
            for name, dq in self._stages.items():
                lat = list(dq)
                stages[name] = {"avg_s": _avg(lat), "p95_s": _p95(lat), "n": len(lat)}
            local = self._backends["local"].as_dict()
            cloud = self._backends["cloud"].as_dict(
                config.CLOUD_PRICE_IN_PER_1M, config.CLOUD_PRICE_OUT_PER_1M)
            fallbacks = self._fallbacks
            health = dict(self._local_health)
        with self._sys_lock:
            gpu = self._gpu_latest
            cpu = self._cpu_latest
            ram = self._ram_latest
            gpu_hist = [x for x in self._gpu_hist if x is not None]
            cpu_hist = [x for x in self._cpu_hist if x is not None]
            ram_hist = [x for x in self._ram_hist if x is not None]
            m4 = dict(self._m4)
            m4_gpu_hist = [x for x in self._m4_gpu_hist if x is not None]
            m4_ram_hist = [x for x in self._m4_ram_hist if x is not None]
            m4_llm_rss_hist = [x for x in self._m4_llm_rss_hist if x is not None]
        return {
            "uptime": f"{up // 3600}h {(up % 3600) // 60}m",
            "turns": self.turns,
            "avg_turn_latency_s": stages.get("turn", {}).get("avg_s"),
            "stages": stages,
            "llm": {"local": local, "cloud": cloud, "fallbacks": fallbacks,
                    "local_health": health},
            "gpu": gpu, "cpu": cpu, "ram": ram,
            "m4": m4,
            "history": {"gpu": gpu_hist, "cpu": cpu_hist, "ram": ram_hist,
                        "m4_gpu": m4_gpu_hist, "m4_ram": m4_ram_hist,
                        "m4_llm_rss": m4_llm_rss_hist},
            "top_tools": [{"name": n, "count": c} for n, c in top],
            # kept for backward-compat with the existing panel
            "vram": (f"{gpu['mem_used']}/{gpu['mem_total']} MiB" if gpu else None),
        }


class SystemSampler(threading.Thread):
    """Polls GPU (nvidia-smi), CPU + RAM (psutil) on an interval into Metrics."""

    def __init__(self, metrics: Metrics, interval_s: float, stop_event=None) -> None:
        super().__init__(daemon=True, name="metrics-sampler")
        self.metrics = metrics
        self.interval = max(1.0, interval_s)
        self._stop = stop_event
        self._pid = os.getpid()
        self._proc = psutil.Process(self._pid) if psutil else None
        if psutil:  # prime the non-blocking cpu_percent counters
            try:
                psutil.cpu_percent(percpu=True)
                self._proc.cpu_percent()
            except Exception:
                pass

    def _sample_gpu(self) -> dict | None:
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip().splitlines()[0]
            util, mused, mtot, temp = [x.strip() for x in out.split(",")]
            return {"util": float(util), "mem_used": int(float(mused)),
                    "mem_total": int(float(mtot)),
                    "mem_percent": round(float(mused) / float(mtot) * 100, 1),
                    "temp_c": float(temp)}
        except Exception:
            return None

    def _sample_cpu(self) -> dict | None:
        if not psutil:
            return None
        try:
            per = psutil.cpu_percent(percpu=True)
            proc = None
            try:
                proc = round(self._proc.cpu_percent(), 1)
            except Exception:
                pass
            return {"overall": round(sum(per) / len(per), 1) if per else None,
                    "per_core": [round(x, 1) for x in per],
                    "cores": len(per), "mark_proc_pct": proc}
        except Exception:
            return None

    def _sample_ram(self) -> dict | None:
        if not psutil:
            return None
        try:
            vm = psutil.virtual_memory()
            rss = None
            try:
                rss = round(self._proc.memory_info().rss / 1073741824, 2)
            except Exception:
                pass
            return {"percent": vm.percent,
                    "used_gb": round(vm.used / 1073741824, 1),
                    "total_gb": round(vm.total / 1073741824, 1),
                    "mark_rss_gb": rss}
        except Exception:
            return None

    def run(self) -> None:
        while True:
            if self._stop is not None and self._stop.is_set():
                return
            try:
                self.metrics.push_system_sample(
                    self._sample_gpu(), self._sample_cpu(), self._sample_ram())
            except Exception:
                logger.exception("system sample failed")
            time.sleep(self.interval)

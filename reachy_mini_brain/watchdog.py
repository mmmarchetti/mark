"""Auto-recovery watchdog for the daemon's motor-communication fault.

After long uptime the daemon can hit a transient "Motor communication error"
(state:"error") - the robot keeps answering (audio works) but won't move,
head stuck. A plain app restart can't fix a faulted-but-running daemon; only a
full daemon cycle does. This watchdog detects that state and runs
`markctl recover` (which cycles daemon + app), so the fault self-heals.
"""

import logging
import subprocess
import threading
import time

import requests

from reachy_mini_brain import config

logger = logging.getLogger(__name__)


class RecoveryWatchdog:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="watchdog")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @staticmethod
    def _self_rss_mb() -> float:
        """This process's resident set size, in MB (0.0 if unreadable)."""
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _daemon_faulted() -> bool:
        try:
            r = requests.get("http://localhost:8000/api/daemon/status", timeout=3)
            d = r.json()
        except Exception:
            return False  # unreachable != faulted; could be mid-restart
        if d.get("state") == "error":
            return True
        b = d.get("backend_status") or {}
        if b.get("error"):
            return True
        return False

    def _run(self) -> None:
        time.sleep(config.WATCHDOG_STARTUP_DELAY_S)
        logger.info("Recovery watchdog started.")
        bad_since = 0.0
        mem_bad_since = 0.0
        while not self._stop.is_set():
            time.sleep(config.WATCHDOG_POLL_S)

            # Memory self-guard: catch a runaway leak and cycle gracefully
            # before the OS OOM-killer nukes the process (and the desktop).
            if config.MEMORY_GUARD_ENABLED:
                rss = self._self_rss_mb()
                if rss >= config.MEMORY_GUARD_MB:
                    now = time.time()
                    if mem_bad_since == 0.0:
                        mem_bad_since = now
                        logger.warning(
                            "Memory high: RSS %.0f MB >= guard %.0f MB - watching.",
                            rss, config.MEMORY_GUARD_MB,
                        )
                    elif now - mem_bad_since >= config.MEMORY_GUARD_CONFIRM_S:
                        logger.error(
                            "Memory leak guard: RSS %.0f MB sustained; running "
                            "'markctl recover' before OOM.", rss,
                        )
                        try:
                            subprocess.Popen(
                                ["bash", "-lc", "markctl recover"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            )
                        except Exception:
                            logger.exception("Failed to invoke markctl recover (memory)")
                        return  # recover replaces this process
                else:
                    mem_bad_since = 0.0

            if self._daemon_faulted():
                now = time.time()
                if bad_since == 0.0:
                    bad_since = now
                    logger.warning("Daemon fault detected - watching before recovery.")
                elif now - bad_since >= config.WATCHDOG_CONFIRM_S:
                    logger.error("Daemon fault persisted; running 'markctl recover'.")
                    try:
                        # This cycles the daemon AND this app - our process will
                        # be replaced, which is the intended full recovery.
                        subprocess.Popen(
                            ["bash", "-lc", "markctl recover"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                    except Exception:
                        logger.exception("Failed to invoke markctl recover")
                    return  # stop watching; recover will restart everything
            else:
                bad_since = 0.0
        logger.info("Recovery watchdog stopped.")

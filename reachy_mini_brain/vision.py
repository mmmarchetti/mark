"""On-demand vision via local Qwen3-VL (Ollama, GPU)."""

import base64
import io
import logging

import numpy as np
import requests
from PIL import Image

from reachy_mini_brain import config

logger = logging.getLogger(__name__)

# Smaller frame => fewer image tokens => much faster prompt processing. 512 is
# plenty for "what/who is in front of me" and roughly halves vision latency vs
# 768 (camera turns were running 8-14s live). Override via REACHY_VISION_MAX_DIM.
MAX_DIMENSION = int(config.VISION_MAX_DIMENSION)


def _frame_to_base64_jpeg(frame: np.ndarray) -> str:
    image = Image.fromarray(frame)
    image.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VisionTool:
    """Captures a camera frame and asks the local VLM to describe it."""

    def __init__(self) -> None:
        self.endpoint = f"{config.OLLAMA_ENDPOINT}/api/generate"
        self.model = config.VISION_MODEL

    def describe(self, reachy_mini, question: str) -> str:
        """Capture the current frame and describe it."""
        frame = reachy_mini.media.get_frame()
        if frame is None:
            return "I couldn't access the camera right now."
        return self.describe_frames([frame], question)

    def describe_frames(self, frames: list[np.ndarray], question: str) -> str:
        """Describe one or more already-captured frames in a SINGLE request.

        Sending every frame in one call (Ollama's /api/generate takes an
        `images` array) means a whole body scan costs one inference instead of
        one per angle - which is what made the scan stop-and-go, since each
        per-angle inference blocked the sweep for seconds.
        """
        if not frames:
            return "I couldn't access the camera right now."

        # More images means more image tokens, so give the context room.
        num_ctx = 4096 if len(frames) <= 1 else 8192
        # Ask for a brief description AND cap output tokens: the model was
        # generating long paragraphs, which is both slow to generate and too
        # verbose to speak. The brain re-summarizes anyway.
        prompt = question.strip() + "\nAnswer in one or two short sentences."
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [_frame_to_base64_jpeg(f) for f in frames],
            "stream": False,
            "options": {"num_ctx": num_ctx, "num_predict": config.VISION_MAX_TOKENS},
            # keep_alive:0 forced a full ~6GB reload on every single call,
            # sometimes exceeding even a 30s timeout - confirmed live
            # (ReadTimeoutError). Keep it warm for a few minutes instead;
            # trades some VRAM headroom for much better conversational
            # latency, and it still unloads on its own once idle.
            "keep_alive": config.VISION_KEEP_ALIVE,
        }
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=config.VISION_TIMEOUT_S)
            resp.raise_for_status()
            return resp.json().get("response", "").strip() or "I couldn't make out anything useful."
        except requests.RequestException:
            logger.exception("Vision request to Ollama failed")
            return "I had trouble looking right now."

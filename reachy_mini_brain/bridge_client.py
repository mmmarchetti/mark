"""HTTP client to the Mark Bridge running on the MacBook (over Tailscale).

The Bridge owns the Slack + Google Calendar MCP servers and credentials; Mark
only makes short HTTP calls to it. Everything degrades gracefully: if the URL
isn't configured or the Mac is offline/asleep, calls return a friendly message
(same philosophy as vision.py) so the robot never hangs or crashes.

Failure modes are kept DISTINCT so Mark never misleads the user:
  * URL unset            -> _UNSET_MSG
  * connect/timeout      -> _OFFLINE_MSG   (Mac truly unreachable)
  * HTTP error response  -> service message (Mac WAS reached; the service failed)
The last case matters: an expired Google Calendar token makes the bridge answer
HTTP 500 - the Mac is perfectly reachable, so saying "I can't reach your
MacBook" was wrong and sent us chasing network ghosts.
"""

import logging

import requests

from reachy_mini_brain import config

logger = logging.getLogger(__name__)

# Truly could not reach the Mac (DNS/connect/timeout - Tailscale down, Mac
# asleep, or the bridge process not running).
_OFFLINE_MSG = "I can't reach your MacBook right now, so I can't check that."
# The bridge URL isn't configured at all.
_UNSET_MSG = "That isn't set up yet - the MacBook bridge hasn't been configured."
# We DID reach the Mac, but the bridge/service answered with an error status.
_SERVICE_MSG = "I reached your MacBook, but that service ran into an error."
# The Google Calendar OAuth token expired / was revoked and needs re-authorizing.
_CAL_AUTH_MSG = (
    "I reached your MacBook, but its calendar sign-in has expired. "
    "It needs to be re-authorized before I can check your calendar."
)


def _describe_http_error(exc: requests.HTTPError) -> str:
    """Turn a bridge HTTP error into a spoken-friendly, ACCURATE message.

    The Mac answered with a status code, so it is NOT offline - never say so.
    Pull the server's `detail` and special-case the expired-calendar-token
    signature (RefreshError: invalid_grant) so the user hears the real,
    actionable cause instead of a generic error.
    """
    detail = ""
    resp = exc.response
    if resp is not None:
        try:
            detail = str((resp.json() or {}).get("detail", "")) or (resp.text or "")
        except ValueError:
            detail = resp.text or ""
    low = detail.lower()
    if any(k in low for k in ("invalid_grant", "refresherror",
                              "token has been expired", "been revoked")):
        logger.warning("Bridge calendar token needs re-auth: %s", detail)
        return _CAL_AUTH_MSG
    logger.warning("Bridge returned HTTP error: %s", detail or exc)
    return _SERVICE_MSG


class BridgeClient:
    def __init__(self) -> None:
        self.base = config.BRIDGE_URL.rstrip("/")
        self.secret = config.BRIDGE_SECRET
        self.timeout = config.BRIDGE_TIMEOUT_S

    @property
    def configured(self) -> bool:
        return bool(self.base)

    def _headers(self) -> dict:
        return {"X-Mark-Secret": self.secret} if self.secret else {}

    def get(self, path: str, params: dict | None = None):
        if not self.configured:
            return None, _UNSET_MSG
        try:
            r = requests.get(self.base + path, params=params or {},
                             headers=self._headers(), timeout=self.timeout)
            r.raise_for_status()
            return r.json(), None
        except requests.HTTPError as exc:
            return None, _describe_http_error(exc)
        except requests.RequestException:
            logger.warning("Bridge GET %s failed (Mac offline?)", path)
            return None, _OFFLINE_MSG

    def post(self, path: str, body: dict):
        if not self.configured:
            return None, _UNSET_MSG
        try:
            r = requests.post(self.base + path, json=body,
                              headers=self._headers(), timeout=self.timeout)
            r.raise_for_status()
            return r.json(), None
        except requests.HTTPError as exc:
            return None, _describe_http_error(exc)
        except requests.RequestException:
            logger.warning("Bridge POST %s failed (Mac offline?)", path)
            return None, _OFFLINE_MSG

    def health(self) -> bool:
        data, _ = self.get("/health")
        return bool(data and data.get("ok"))

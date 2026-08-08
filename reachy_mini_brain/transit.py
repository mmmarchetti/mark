"""Maps / transit via the Google Maps Directions API. Needs REACHY_MAPS_KEY;
degrades gracefully (like the bridge) when the key isn't set. Saved home/work
addresses let the user ask "how long to work?" without repeating an address.
"""

import logging

import requests

from reachy_mini_brain import config

logger = logging.getLogger(__name__)

_URL = "https://maps.googleapis.com/maps/api/directions/json"
_MODES = {"driving", "walking", "bicycling", "transit"}


class Transit:
    def __init__(self) -> None:
        self.key = config.MAPS_KEY

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def _resolve(self, place: str) -> str:
        """Map the words 'home'/'work' to the saved addresses."""
        p = (place or "").strip()
        low = p.lower()
        if low in ("home", "casa") and config.HOME_ADDRESS:
            return config.HOME_ADDRESS
        if low in ("work", "office", "trabalho", "escritorio", "escritório") and config.WORK_ADDRESS:
            return config.WORK_ADDRESS
        return p

    def directions(self, destination: str, origin: str | None = None, mode: str = "driving") -> str:
        if not self.configured:
            return ("Maps aren't set up yet. Add a Google Maps API key as REACHY_MAPS_KEY "
                    "to enable directions and commute times.")
        mode = mode if mode in _MODES else "driving"
        dest = self._resolve(destination)
        orig = self._resolve(origin) if origin else config.HOME_ADDRESS
        if not orig:
            return "I don't know where to start from. Tell me an origin, or set your home address."
        if not dest:
            return "Where do you want to go?"
        try:
            r = requests.get(_URL, params={
                "origin": orig, "destination": dest, "mode": mode,
                "departure_time": "now", "key": self.key,
            }, timeout=10)
            data = r.json()
            status = data.get("status")
            if status != "OK" or not data.get("routes"):
                logger.warning("Directions status=%s", status)
                if status == "ZERO_RESULTS":
                    return f"I couldn't find a {mode} route to {dest}."
                if status in ("REQUEST_DENIED", "INVALID_REQUEST"):
                    return "Maps rejected the request - the API key may be missing Directions access."
                return "I couldn't get directions right now."
            leg = data["routes"][0]["legs"][0]
            dur = leg.get("duration_in_traffic", leg.get("duration", {})).get("text", "?")
            dist = leg.get("distance", {}).get("text", "")
            how = {"driving": "driving", "walking": "on foot",
                   "bicycling": "by bike", "transit": "by public transit"}[mode]
            tail = f" ({dist})" if dist else ""
            return f"About {dur} {how} to {leg.get('end_address', dest)}{tail}."
        except Exception:
            logger.exception("directions failed")
            return "I couldn't reach the maps service right now."

"""Weather via Open-Meteo (free, no API key) with IP-based geolocation.

Location is auto-detected once from the machine's IP and cached; the user can
also ask about a named city, which we geocode on demand. Weather codes are
mapped to short human descriptions.
"""

import logging
import threading

import requests

from reachy_mini_brain import config

logger = logging.getLogger(__name__)

# WMO weather interpretation codes -> short description.
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "rime fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm",
}


class Weather:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._home: dict | None = None  # cached IP-based location

    def _home_location(self) -> dict | None:
        with self._lock:
            if self._home is not None:
                return self._home
        loc = None
        # Allow a manual override via env, else detect from IP.
        if config.WEATHER_LAT and config.WEATHER_LON:
            loc = {"lat": float(config.WEATHER_LAT), "lon": float(config.WEATHER_LON),
                   "name": config.WEATHER_PLACE or "your location",
                   "timezone": config.WEATHER_TIMEZONE or "auto"}
        else:
            try:
                r = requests.get("http://ip-api.com/json/", timeout=6).json()
                if r.get("status") == "success":
                    loc = {"lat": r["lat"], "lon": r["lon"],
                           "name": f"{r.get('city','')}".strip() or "your location",
                           "timezone": r.get("timezone", "auto")}
            except Exception:
                logger.exception("IP geolocation failed")
        with self._lock:
            self._home = loc
        return loc

    def _geocode(self, place: str) -> dict | None:
        try:
            r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                             params={"name": place, "count": 1}, timeout=6).json()
            res = (r.get("results") or [None])[0]
            if res:
                return {"lat": res["latitude"], "lon": res["longitude"],
                        "name": res.get("name", place), "timezone": res.get("timezone", "auto")}
        except Exception:
            logger.exception("Geocoding failed for %r", place)
        return None

    def describe(self, place: str | None = None) -> str:
        loc = self._geocode(place) if place else self._home_location()
        if loc is None:
            return "I couldn't figure out the location for the weather right now."
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": loc["lat"], "longitude": loc["lon"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
                "timezone": loc["timezone"], "forecast_days": 1,
            }, timeout=8).json()
        except Exception:
            logger.exception("Weather fetch failed")
            return "I couldn't reach the weather service right now."

        cur = r.get("current", {})
        day = r.get("daily", {})
        desc = _WMO.get(cur.get("weather_code"), "")
        temp = round(cur.get("temperature_2m", 0))
        feels = round(cur.get("apparent_temperature", temp))
        hi = round(day.get("temperature_2m_max", [temp])[0])
        lo = round(day.get("temperature_2m_min", [temp])[0])
        rain = day.get("precipitation_probability_max", [None])[0]

        # Plain words only - no °, C, % symbols (they are converted straight to
        # speech and mispronounce). The brain rephrases this in the user's language.
        parts = [f"In {loc['name']} it's {temp} degrees Celsius"]
        if abs(feels - temp) >= 2:
            parts[0] += f" (feels like {feels} degrees)"
        if desc:
            parts.append(desc)
        parts.append(f"today {lo} to {hi} degrees")
        if rain is not None:
            parts.append(f"{rain} percent chance of rain")
        return ", ".join(parts) + "."

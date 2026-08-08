"""Stocks, crypto, and currency - all via keyless public endpoints.

- stocks: Stooq CSV (e.g. aapl.us)
- crypto: CoinGecko simple price
- currency: open.er-api.com
Returns short plain-text (numbers spoken as words downstream by tts).
"""

import logging

import requests

logger = logging.getLogger(__name__)


class Finance:
    def stock(self, symbol: str) -> str:
        sym = (symbol or "").strip().lower()
        if not sym:
            return "Which stock?"
        s = sym if "." in sym else f"{sym}.us"
        try:
            txt = requests.get(f"https://stooq.com/q/l/?s={s}&f=sd2t2ohlcv&h&e=csv", timeout=6).text
            rows = txt.strip().splitlines()
            if len(rows) < 2 or "N/D" in rows[1]:
                return f"I couldn't find a price for {symbol}."
            cols = rows[1].split(",")
            close = cols[6]
            return f"{symbol.upper()} is trading at {close} dollars."
        except Exception:
            logger.exception("stock lookup failed")
            return "I couldn't get that stock price."

    def crypto(self, coin: str, vs: str = "usd") -> str:
        c = (coin or "").strip().lower()
        aliases = {"btc": "bitcoin", "eth": "ethereum", "sol": "solana", "doge": "dogecoin"}
        c = aliases.get(c, c)
        if not c:
            return "Which coin?"
        try:
            r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                             params={"ids": c, "vs_currencies": vs}, timeout=6).json()
            price = (r.get(c) or {}).get(vs)
            if price is None:
                return f"I couldn't find a price for {coin}."
            return f"{c.capitalize()} is {price} {vs.upper()}."
        except Exception:
            logger.exception("crypto lookup failed")
            return "I couldn't get that crypto price."

    def currency(self, amount: float, frm: str, to: str) -> str:
        frm, to = (frm or "").upper().strip(), (to or "").upper().strip()
        if not frm or not to:
            return "From which currency to which?"
        try:
            r = requests.get(f"https://open.er-api.com/v6/latest/{frm}", timeout=6).json()
            rate = (r.get("rates") or {}).get(to)
            if rate is None:
                return f"I couldn't convert {frm} to {to}."
            return f"{amount} {frm} is about {round(amount * rate, 2)} {to}."
        except Exception:
            logger.exception("currency lookup failed")
            return "I couldn't do that conversion right now."

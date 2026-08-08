"""Wikipedia, finance (stocks/crypto/currency), and news tools."""

from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies


class WikipediaTool(Tool):
    name = "wikipedia"
    description = (
        "Look up factual/encyclopedic information on Wikipedia (people, places, "
        "history, science, definitions). Use for deeper facts than a web snippet. "
        "Summarize briefly in the user's language."
    )
    parameters_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Topic to look up."}},
        "required": ["query"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        if deps.wiki is None:
            return "Wikipedia isn't available right now."
        return deps.wiki.lookup(str(kwargs.get("query", "")), lang=str(kwargs.get("_lang", "en")))


class FinanceTool(Tool):
    name = "finance"
    description = (
        "Get a stock price, crypto price, or convert currency. kind='stock' with symbol "
        "(e.g. AAPL), kind='crypto' with symbol (e.g. BTC), or kind='currency' with "
        "amount, from_ccy and to_ccy (e.g. 100 USD to BRL)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["stock", "crypto", "currency"]},
            "symbol": {"type": "string", "description": "Ticker or coin (stock/crypto)."},
            "amount": {"type": "number", "description": "Amount to convert (currency)."},
            "from_ccy": {"type": "string", "description": "Source currency code (currency)."},
            "to_ccy": {"type": "string", "description": "Target currency code (currency)."},
        },
        "required": ["kind"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        if deps.finance is None:
            return "Finance info isn't available right now."
        kind = kwargs.get("kind")
        if kind == "stock":
            return deps.finance.stock(str(kwargs.get("symbol", "")))
        if kind == "crypto":
            return deps.finance.crypto(str(kwargs.get("symbol", "")))
        if kind == "currency":
            return deps.finance.currency(float(kwargs.get("amount", 1) or 1),
                                         str(kwargs.get("from_ccy", "")), str(kwargs.get("to_ccy", "")))
        return "Tell me a stock, a coin, or a currency conversion."


class NewsTool(Tool):
    name = "news"
    description = (
        "Read the latest news headlines. Optional topic: top, tech, world, business. "
        "Summarize briefly in the user's language."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "enum": ["top", "tech", "world", "business"]},
        },
        "required": [],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        if deps.news is None:
            return "News isn't available right now."
        return deps.news.headlines(str(kwargs.get("topic", "top")),
                                   language=str(kwargs.get("_lang", "pt")))

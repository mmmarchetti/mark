from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies


class SlackRecentTool(Tool):
    """Read recent Slack messages/mentions (via the MacBook bridge)."""

    name = "slack_recent"
    description = (
        "Check the user's recent Slack activity - unread messages, mentions, or DMs - "
        "and summarize it. Use when they ask if there's anything new on Slack, or to "
        "catch them up. Read-only. Optionally pass a channel name to focus on one."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "channel": {"type": "string", "description": "Optional channel/DM name to focus on."},
            "limit": {"type": "integer", "description": "Max messages to fetch (default 10)."},
        },
        "required": [],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        if deps.bridge is None:
            return "Slack isn't available right now."
        params = {"limit": int(kwargs.get("limit", 10) or 10)}
        if kwargs.get("channel"):
            params["channel"] = str(kwargs["channel"]).strip()
        data, err = deps.bridge.get("/slack/recent", params)
        if err:
            return err
        msgs = data.get("messages", [])
        if not msgs:
            return "Nothing new on Slack."
        parts = []
        for m in msgs[:10]:
            who = m.get("from", "someone")
            where = m.get("channel", "")
            text = (m.get("text", "") or "").replace("\n", " ")
            if len(text) > 160:
                text = text[:157] + "..."
            parts.append(f"{who} in {where}: {text}" if where else f"{who}: {text}")
        return "Recent Slack: " + " | ".join(parts)


class SlackSendTool(Tool):
    """Send a Slack message (write - needs confirmation)."""

    name = "slack_send"
    description = (
        "Send a message to a Slack channel or person. This POSTS on the user's behalf, "
        "so first tell them exactly what you'll send and to whom, and ask them to "
        "confirm; only call this with confirmed=true after they clearly say yes."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "channel": {"type": "string", "description": "Channel name (e.g. #general) or person to send to."},
            "text": {"type": "string", "description": "The message text."},
            "confirmed": {"type": "boolean", "description": "Must be true; only set after the user confirms."},
        },
        "required": ["channel", "text", "confirmed"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        if not kwargs.get("confirmed"):
            return "Not confirmed yet. Tell the user the exact message and recipient and ask them to confirm first."
        if deps.bridge is None:
            return "Slack isn't available right now."
        body = {
            "channel": str(kwargs.get("channel", "")).strip(),
            "text": str(kwargs.get("text", "")).strip(),
        }
        data, err = deps.bridge.post("/slack/send", body)
        if err:
            return err
        return "Sent." if data.get("ok") else "I couldn't send that message."

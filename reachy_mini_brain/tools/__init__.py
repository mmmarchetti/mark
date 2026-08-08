"""Tool definitions the LLM can call. Motion tools push onto a queue drained
by the SDK control loop, per reachy_mini/skills/ai-integration.md: the LLM
never touches motors directly, keeping motion smooth regardless of LLM
latency.
"""

from reachy_mini_brain.tools.core import ToolDependencies, ToolRegistry
from reachy_mini_brain.tools.calendar import CalendarAgendaTool, CalendarCreateTool
from reachy_mini_brain.tools.camera import CameraTool
from reachy_mini_brain.tools.dance import DanceTool
from reachy_mini_brain.tools.desk import DeskTool
from reachy_mini_brain.tools.go_to_sleep import GoToSleepTool
from reachy_mini_brain.tools.head_tracking import HeadTrackingTool
from reachy_mini_brain.tools.identity import EnrollIdentityTool
from reachy_mini_brain.tools.game import PlayGameTool
from reachy_mini_brain.tools.info import FinanceTool, NewsTool, WikipediaTool
from reachy_mini_brain.tools.look_around import LookAroundTool
from reachy_mini_brain.tools.memo import RecordMemoTool
from reachy_mini_brain.tools.memory import ForgetTool, RememberTool
from reachy_mini_brain.tools.move_head import MoveHeadTool
from reachy_mini_brain.tools.personality import SetPersonalityTool
from reachy_mini_brain.tools.play_emotion import PlayEmotionTool
from reachy_mini_brain.tools.pomodoro import PomodoroTool
from reachy_mini_brain.tools.react import ReactTool
from reachy_mini_brain.tools.reminder import RemindersTool, SetReminderTool
from reachy_mini_brain.tools.stop_listening import StopListeningTool
from reachy_mini_brain.tools.todo import TodoTool
from reachy_mini_brain.tools.transit import TransitTool
from reachy_mini_brain.tools.weather import WeatherTool
from reachy_mini_brain.tools.websearch import WebSearchTool


def build_default_registry(available_moves: list[str] | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    # PlayEmotionTool needs the real move list so it can expose them as an
    # enum rather than a few names buried in prose.
    registry.register(PlayEmotionTool(available_moves))
    for tool_cls in (
        MoveHeadTool,
        DanceTool,
        ReactTool,
        RecordMemoTool,
        CameraTool,
        LookAroundTool,
        HeadTrackingTool,
        EnrollIdentityTool,
        PlayGameTool,
        WeatherTool,
        WebSearchTool,
        WikipediaTool,
        FinanceTool,
        NewsTool,
        TransitTool,
        SetReminderTool,
        RemindersTool,
        TodoTool,
        PomodoroTool,
        CalendarAgendaTool,
        CalendarCreateTool,
        RememberTool,
        ForgetTool,
        SetPersonalityTool,
        DeskTool,
        GoToSleepTool,
        StopListeningTool,
    ):
        registry.register(tool_cls())
    return registry


__all__ = ["ToolDependencies", "ToolRegistry", "build_default_registry"]

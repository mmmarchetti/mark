from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies


class EnrollIdentityTool(Tool):
    """Enroll / list / forget the people Mark recognizes by face and voice."""

    name = "identity"
    description = (
        "Manage who you recognize by face and voice. Actions: "
        "'enroll' - learn the CURRENT person (captures their face from the camera "
        "and their voice from what they just said); requires 'name'. Use when "
        "someone says things like 'remember me, I'm Marcos' or 'learn my face'. "
        "Make sure they are looking at the camera and have just spoken. "
        "'list' - say who you currently recognize. "
        "'forget' - stop recognizing a person; requires 'name'. "
        "After enrolling, you'll then greet that person by name automatically."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["enroll", "list", "forget"]},
            "name": {"type": "string", "description": "Person's name (for enroll/forget)."},
        },
        "required": ["action"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        idn = deps.identity
        if idn is None:
            return "Recognition isn't available right now."
        action = str(kwargs.get("action", "")).strip().lower()

        if action == "list":
            names = idn.names()
            return "I recognize " + ", ".join(names) + "." if names else "I don't recognize anyone yet."

        if action == "forget":
            name = str(kwargs.get("name", "")).strip()
            if not name:
                return "Whose face and voice should I forget?"
            return idn.forget(name)

        if action == "enroll":
            name = str(kwargs.get("name", "")).strip()
            if not name:
                return "What's the person's name so I can remember them?"
            face_emb = voice_emb = None
            try:
                frame = deps.reachy_mini.media.get_frame()
                if frame is not None:
                    face_emb = idn.face_embedding(frame)
            except Exception:
                pass
            listener = deps.listener
            audio = getattr(listener, "last_utterance", None) if listener else None
            if audio is not None:
                voice_emb = idn.voice_embedding(audio)
            if face_emb is None and voice_emb is None:
                return (f"I couldn't capture a face or voice sample for {name}. "
                        f"Ask them to look at the camera and say a sentence, then try again.")
            return idn.enroll(name, face_emb=face_emb, voice_emb=voice_emb)

        return "Unknown identity action."

from typing import List, Literal, Optional, Protocol, TypedDict

from .decision import (
    SupportDecision,
    assess_customer_message,
    decide_support_action,
)

Role = Literal["system", "user", "assistant"]
ResponseStyle = Literal["neutral", "warm", "direct"]
SupportTask = Literal["general", "troubleshooting", "policy", "recommendation"]

_RESPONSE_STYLE_GUIDANCE = {
    "neutral": "Use a neutral, professional tone.",
    "warm": (
        "Use a warm, calm tone while staying honest; do not pretend to feel "
        "emotions or mirror distress."
    ),
    "direct": "Use a direct, concise tone without unnecessary pleasantries.",
}
_TASK_GUIDANCE = {
    "general": "",
    "troubleshooting": (
        "Troubleshooting task: offer one safe, reversible step at a time. Ask for "
        "only details needed to identify the issue and do not claim a diagnosis."
    ),
    "policy": (
        "Policy task: rely on the supplied, current policy. Distinguish what is "
        "known from what is not documented; do not invent exceptions."
    ),
    "recommendation": (
        "Recommendation task: use criteria the customer explicitly gives, ask "
        "permission before collecting optional preferences, and show relevant "
        "trade-offs without steering toward a hidden outcome."
    ),
}


class Message(TypedDict):
    role: Role
    content: str


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]
    logprobs: List[float]


Dialog = List[Message]


class ChatGenerator(Protocol):
    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        ...


_BASE_SERVICE_GUIDANCE = (
    "You are helping with customer service. Be clear, respectful, and concise. "
    "Treat the support cues below as limited signals in the wording, not as facts "
    "about the customer's inner emotional state. Do not infer or use demographic "
    "or psychological traits. Do not pressure the customer, invent policies, or "
    "claim that you took an account action. Ask for only information needed to help. "
    "Only when relevant, ask permission before using preferences the customer "
    "explicitly provides; never infer or persist an interest profile."
)


class CustomerServiceAssistant:
    """Adds optional, transparent support cues to a compatible chat generator."""

    def __init__(
        self,
        generator: ChatGenerator,
        system_prompt: str = _BASE_SERVICE_GUIDANCE,
        response_style: Optional[ResponseStyle] = None,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        if response_style is not None and response_style not in _RESPONSE_STYLE_GUIDANCE:
            raise ValueError("response_style must be neutral, warm, direct, or None")
        self._generator = generator
        self._system_prompt = system_prompt.strip()
        self._response_style = response_style

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        task: SupportTask = "general",
    ) -> List[ChatPrediction]:
        """Generate support replies without changing the caller's dialog objects."""
        if not dialogs:
            raise ValueError("at least one dialog is required")
        if task not in _TASK_GUIDANCE:
            raise ValueError(
                "task must be general, troubleshooting, policy, or recommendation"
            )
        prepared_dialogs = [
            self._prepare_dialog(dialog, task) for dialog in dialogs
        ]
        return self._generator.chat_completion(
            prepared_dialogs,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
            logprobs=logprobs,
        )

    def _prepare_dialog(self, dialog: Dialog, task: SupportTask) -> Dialog:
        if not dialog:
            raise ValueError("each dialog must contain at least one message")
        if dialog[-1]["role"] != "user":
            raise ValueError("each dialog must end with a user message")

        assessment = assess_customer_message(dialog[-1]["content"])
        decision = decide_support_action(assessment)
        instructions = [
            self._system_prompt,
            self._decision_guidance(decision),
        ]
        if _TASK_GUIDANCE[task]:
            instructions.append(_TASK_GUIDANCE[task])
        if self._response_style is not None:
            instructions.append(
                _RESPONSE_STYLE_GUIDANCE[self._response_style]
            )
        system_content = "\n".join(instructions)

        copied_dialog: Dialog = [
            {"role": message["role"], "content": message["content"]}
            for message in dialog
        ]
        if copied_dialog[0]["role"] == "system":
            copied_dialog[0]["content"] = (
                copied_dialog[0]["content"] + "\n\n" + system_content
            )
        else:
            copied_dialog.insert(
                0, {"role": "system", "content": system_content}
            )
        return copied_dialog

    @staticmethod
    def _decision_guidance(decision: SupportDecision) -> str:
        guidance = []
        if decision.acknowledge:
            guidance.append(
                "Briefly acknowledge the stated service problem or feeling without "
                "guessing beyond the customer's words."
            )
        if decision.route == "urgent_human_review":
            guidance.append(
                "Support route: clarify the immediate need and offer timely human "
                "review. Treat urgency wording as a cue, not proof of an emergency; "
                "do not claim a transfer or promise an outcome."
            )
        elif decision.route == "human_support":
            guidance.append(
                "Support route: the customer explicitly requested a person. Explain "
                "the available human-support next step without claiming a transfer."
            )
        else:
            guidance.append(
                "Support route: answer the customer's stated request. Ask a neutral "
                "clarifying question only if needed."
            )
        return " ".join(guidance)

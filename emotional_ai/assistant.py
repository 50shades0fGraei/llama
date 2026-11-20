from typing import List, Literal, Optional, Protocol, TypedDict

from .decision import SupportAssessment, assess_customer_message

Role = Literal["system", "user", "assistant"]


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
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        self._generator = generator
        self._system_prompt = system_prompt.strip()

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """Generate support replies without changing the caller's dialog objects."""
        if not dialogs:
            raise ValueError("at least one dialog is required")
        prepared_dialogs = [self._prepare_dialog(dialog) for dialog in dialogs]
        return self._generator.chat_completion(
            prepared_dialogs,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
            logprobs=logprobs,
        )

    def _prepare_dialog(self, dialog: Dialog) -> Dialog:
        if not dialog:
            raise ValueError("each dialog must contain at least one message")
        if dialog[-1]["role"] != "user":
            raise ValueError("each dialog must end with a user message")

        assessment = assess_customer_message(dialog[-1]["content"])
        system_content = self._system_prompt + "\n" + self._assessment_guidance(
            assessment
        )

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
    def _assessment_guidance(assessment: SupportAssessment) -> str:
        guidance = []
        if assessment.acknowledgement_recommended:
            guidance.append(
                "Acknowledge the stated service problem or feeling briefly, "
                "without guessing beyond the customer's words."
            )
        if assessment.human_handoff_requested:
            guidance.append(
                "The customer explicitly requested a person. Explain the available "
                "human-support next step, but do not claim a transfer was completed."
            )
        if assessment.urgent_language:
            guidance.append(
                "The message contains urgency-related wording. Clarify the immediate "
                "need and offer a timely human-support option without claiming a "
                "transfer or promising an outcome."
            )
        if not guidance:
            guidance.append(
                "No specific support cue was detected; answer the customer's request "
                "normally and ask a focused clarifying question if needed."
            )
        return "Support guidance: " + " ".join(guidance)

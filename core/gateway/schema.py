from typing import Any
from pydantic import BaseModel, ConfigDict

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False

class AnthropicMessageRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = 1024
    system: str | list[dict[str, Any]] | None = None
    stream: bool = False
    temperature: float | None = 0.7
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | str | None = None
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")

from typing import Optional, List
from pydantic import BaseModel, Field


# data models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "mock-gpt-model"
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False
    frequency_penalty: Optional[float] = 1.05
    presence_penalty: Optional[float] = 1.05
    top_p: Optional[float] = 0.95

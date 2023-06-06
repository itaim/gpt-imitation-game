from typing import List, Optional

from pydantic.main import BaseModel


class AugmentationRequest(BaseModel):
    emotional_state: str
    memory: List[str]
    search: List[str]
    reply: Optional[str]


class AugmentationResponse(BaseModel):
    memory: List[str]
    search: List[str]


class SessionID(BaseModel):
    username: str
    agent: str
    password: str

    def _first_name(self, fullname:str) -> str:
        # guess from username, remove special characters
        first = self.username.split()[0]
        return first.capitalize()

    def first_name(self)->str:
        return self._first_name(self.username)

    def agent_name(self)->str:
        return self._first_name(self.agent)

# Tuple[Tuple[str, int], Tuple[str, int]]]

class TText(BaseModel):
    text: str
    tokens: int


class ChatEntry(BaseModel):
    user: Optional[TText]
    agent: Optional[TText]

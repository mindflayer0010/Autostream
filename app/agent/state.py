from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from langchain_core.messages import BaseMessage


@dataclass
class AgentState:
    messages: List[BaseMessage] = field(default_factory=list)

    intent: str = ""

    name: Optional[str] = None
    email: Optional[str] = None
    platform: Optional[str] = None

    lead_captured: bool = False

    last_rag_context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": self.messages,
            "intent": self.intent,
            "name": self.name,
            "email": self.email,
            "platform": self.platform,
            "lead_captured": self.lead_captured,
            "last_rag_context": self.last_rag_context,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentState":
        s = cls()
        s.messages = d.get("messages", [])
        s.intent = d.get("intent", "")
        s.name = d.get("name")
        s.email = d.get("email")
        s.platform = d.get("platform")
        s.lead_captured = bool(d.get("lead_captured", False))
        s.last_rag_context = d.get("last_rag_context")
        return s

import os
from dotenv import load_dotenv
load_dotenv()
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.agent.graph import build_agent
from app.agent.state import AgentState

KB_PATH = "app/data/knowledge_base.md"

app = FastAPI(title="AutoStream Agent (ServiceHive Assignment)")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
agent = build_agent(KB_PATH, model=MODEL_NAME)

SESSIONS: Dict[str, AgentState] = {}


class ChatIn(BaseModel):
    session_id: str
    message: str


class ChatOut(BaseModel):
    session_id: str
    reply: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn):
    state = SESSIONS.get(payload.session_id, AgentState())

    state.messages.append(HumanMessage(content=payload.message))
    out = agent.invoke(state.to_dict())
    state = AgentState.from_dict(out)

    SESSIONS[payload.session_id] = state

    reply = next((m.content for m in reversed(state.messages) if m.type == "ai"), "")
    return ChatOut(session_id=payload.session_id, reply=reply)

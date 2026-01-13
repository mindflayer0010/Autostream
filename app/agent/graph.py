from __future__ import annotations

from typing import Dict, Any, Literal

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from .state import AgentState
from .intent import classify_intent
from .rag import build_vectorstore, retrieve
from .extract import extract_fields
from .tools import mock_lead_capture


def build_agent(kb_path: str, model: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model, temperature=0)
    vectorstore = build_vectorstore(kb_path)

    def _latest_user_text(state: AgentState) -> str:
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return ""

    def classify_node(d: Dict[str, Any]) -> Dict[str, Any]:
        s = AgentState.from_dict(d)
        user_text = _latest_user_text(s)
        s.intent = classify_intent(llm, user_text)
        return s.to_dict()

    def greeting_node(d: Dict[str, Any]) -> Dict[str, Any]:
        s = AgentState.from_dict(d)
        s.messages.append(
            AIMessage(
                content=(
                    "Hey! 👋 I’m AutoStream’s assistant. "
                    "Ask me about pricing, plans, features, refunds, or support."
                )
            )
        )
        return s.to_dict()

    def rag_node(d: Dict[str, Any]) -> Dict[str, Any]:
        s = AgentState.from_dict(d)
        user_text = _latest_user_text(s)

        context, _chunks = retrieve(vectorstore, user_text, k=4)
        s.last_rag_context = context

        prompt = (
            "You are AutoStream's product assistant.\n"
            "Answer ONLY using the provided context.\n"
            "If the context does not contain the answer, say: \"I don't know based on the current info.\" \n\n"
            f"Context:\n{context}\n\n"
            f"User question: {user_text}\n\n"
            "Write a short, accurate answer in 2–6 sentences."
        )
        answer = llm.invoke([HumanMessage(content=prompt)]).content
        s.messages.append(AIMessage(content=answer))
        return s.to_dict()

    def sales_node(d: Dict[str, Any]) -> Dict[str, Any]:
        s = AgentState.from_dict(d)
        user_text = _latest_user_text(s)

        extracted = extract_fields(llm, user_text)
        if extracted.name and not s.name:
            s.name = extracted.name
        if extracted.email and not s.email:
            s.email = extracted.email
        if extracted.platform and not s.platform:
            s.platform = extracted.platform

        if not s.name:
            s.messages.append(AIMessage(content="Awesome — what’s your name?"))
            return s.to_dict()

        if not s.email:
            s.messages.append(AIMessage(content="Thanks! What’s your email?"))
            return s.to_dict()

        if not s.platform:
            s.messages.append(
                AIMessage(content="Got it. Which creator platform do you use? (YouTube, Instagram, etc.)")
            )
            return s.to_dict()

        if not s.lead_captured:
            mock_lead_capture(s.name, s.email, s.platform)
            s.lead_captured = True
            s.messages.append(
                AIMessage(
                    content=(
                        "Perfect ✅ I’ve captured your details.\n"
                        f"Name: {s.name}\n"
                        f"Email: {s.email}\n"
                        f"Platform: {s.platform}"
                    )
                )
            )
        else:
            s.messages.append(AIMessage(content="Your details are already captured ✅ Anything else you want to ask?"))

        return s.to_dict()

    def route(d: Dict[str, Any]) -> Literal["greet", "rag", "sales"]:
        s = AgentState.from_dict(d)
        if s.intent == "greeting":
            return "greet"
        if s.intent == "high_intent_lead":
            return "sales"
        return "rag"

    g = StateGraph(dict)
    g.add_node("classify", classify_node)
    g.add_node("greet", greeting_node)
    g.add_node("rag", rag_node)
    g.add_node("sales", sales_node)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route, {"greet": "greet", "rag": "rag", "sales": "sales"})

    g.add_edge("greet", END)
    g.add_edge("rag", END)
    g.add_edge("sales", END)

    return g.compile()

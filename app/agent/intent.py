from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class IntentOut(BaseModel):
    intent: str = Field(description="One of: greeting, product_inquiry, high_intent_lead")


_ALLOWED = {"greeting", "product_inquiry", "high_intent_lead"}


def classify_intent(llm: ChatOpenAI, user_text: str) -> str:
    prompt = (
        "You are an intent classifier for a SaaS product assistant.\n"
        "Classify the user's message into exactly ONE of these labels:\n"
        "- greeting\n"
        "- product_inquiry\n"
        "- high_intent_lead\n\n"
        "Rules:\n"
        "- If the user expresses readiness to sign up, try a plan, or says they want Pro for a platform/channel => high_intent_lead\n"
        "- If they ask about pricing, plans, features, refunds, or support => product_inquiry\n"
        "- If it's only a casual greeting => greeting\n\n"
        "Return ONLY JSON with key 'intent'.\n\n"
        f"User message: {user_text}"
    )

    try:
        out = llm.with_structured_output(IntentOut).invoke([HumanMessage(content=prompt)])
        intent = (out.intent or "").strip()
        if intent in _ALLOWED:
            return intent
    except Exception:
        pass

    t = user_text.lower()
    if any(k in t for k in ["sign up", "signup", "buy", "purchase", "trial", "try", "i want", "i'd like", "pro plan", "subscribe"]):
        return "high_intent_lead"
    if any(k in t for k in ["price", "pricing", "plan", "plans", "cost", "features", "refund", "support", "4k", "captions"]):
        return "product_inquiry"
    if any(k in t for k in ["hi", "hello", "hey", "good morning", "good evening"]):
        return "greeting"
    return "product_inquiry"

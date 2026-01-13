from __future__ import annotations

import re
from typing import Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")


class LeadFields(BaseModel):
    name: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)
    platform: Optional[str] = Field(default=None)


def extract_fields(llm: ChatOpenAI, user_text: str) -> LeadFields:
    email = None
    m = EMAIL_RE.search(user_text)
    if m:
        email = m.group(1)

    prompt = (
        "Extract lead details from the user's message.\n"
        "Return ONLY JSON with keys: name, email, platform.\n"
        "- If a field is not present, return null.\n"
        "- Platform should be like: YouTube, Instagram, TikTok, Twitch, LinkedIn, Podcast, Other.\n\n"
        f"User message: {user_text}"
    )

    try:
        out = llm.with_structured_output(LeadFields).invoke([HumanMessage(content=prompt)])
    except Exception:
        out = LeadFields()

    if email:
        out.email = email

    if out.platform:
        out.platform = out.platform.strip()
        if len(out.platform) > 40:
            out.platform = out.platform[:40]

    if out.name:
        out.name = out.name.strip()
        if len(out.name) > 60:
            out.name = out.name[:60]

    return out

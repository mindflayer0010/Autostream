import os
import random

EXIT_MESSAGES = [
    "\nAgent: Thanks for chatting! Remember, your content deserves the best tools. Have a great day! 🚀",
    "\nAgent: Great speaking with you! If you ever need to level up your video workflow, AutoStream is here. 👋",
    "\nAgent: Catch you later! Don't forget—Pro users save 10+ hours a month. Just saying! 😉",
    "\nAgent: Bye for now! We're ready when you're ready to automate your growth. ✨",
    "\nAgent: Signing off! Keep creating amazing content. We'll be here to handle the rest. 🎬"
]

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage

from app.agent.graph import build_agent
from app.agent.state import AgentState

KB_PATH = "app/data/knowledge_base.md"


def main():
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    agent = build_agent(KB_PATH, model=model)

    state = AgentState()
    print("AutoStream CLI (type 'exit' to quit)")

    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        state.messages.append(HumanMessage(content=user))
        out = agent.invoke(state.to_dict())
        state = AgentState.from_dict(out)

        reply = next((m.content for m in reversed(state.messages) if m.type == "ai"), "")
        print(f"\nAgent: {reply}")

    print(random.choice(EXIT_MESSAGES))


if __name__ == "__main__":
    main()

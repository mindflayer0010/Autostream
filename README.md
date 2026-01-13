# AutoStream Agent

An AI-powered platform that converts social media conversations into qualified business leads. Built for the ServiceHive Assignment.

## How to Run

### Prerequisities
- Python 3.9+
- OpenAI API Key

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment:
   Create a `.env` file with your keys:
   ```
   OPENAI_API_KEY=sk-...
   # (CAN'T REVEAL API KEY FOR SAFETY REASONS, AS IT'S PRIVATE)
   MODEL_NAME=gpt-4o-mini
   ```

### Running the Agent

**Using Docker (Recommended)**:
```bash
docker-compose up --build
```
This will start the API at http://localhost:8000.

**Manual Setup**:
```bash
python -m app.cli
```

**FastAPI Server**:
```bash
uvicorn app.main:app --reload
```
Swagger Docs available at: http://localhost:8000/docs

## Architecture Explanation

### Why LangGraph?
I chose **LangGraph** over AutoGen for this assignment because it provides:
1.  **Fine-grained Control**: We need a deterministic flow (Intent -> RAG -> Lead Capture) rather than open-ended multi-agent conversation.
2.  **State Management**: `AgentState` allows us to strictly type our memory (persisting `name`, `email`, `platform` across turns) and implement "human-in-the-loop" logic easily if needed later.
3.  **Cyclic Graphs**: Perfect for the "ask missing details" loop in the sales flow.

### State Management
State is managed using a Pydantic `AgentState` schema passed between graph nodes.
- **Persistence**: For the assignment, we use an in-memory dictionary in `main.py` (`SESSIONS`). In production, this would be replaced with Redis or Postgres `Checkpointer`.
- **Memory**: The `messages` list in the state retains the full conversation history (5-6+ turns) to provide context to the LLM.

## WhatsApp Deployment (Webhooks)

To integrate this agent with WhatsApp (via Meta Business API or Twilio):

1.  **Webhook Endpoint**: The existing `/chat` endpoint in `main.py` is designed for this. It accepts a message and returns a reply.
2.  **Verification**: Add a `GET` endpoint to handle the initial webhook verification challenge from Meta (verifying the `hub.verify_token`).
3.  **Payload Handling**:
    *   Parse the incoming JSON from WhatsApp to extract the user's phone number (acts as `session_id`) and message body.
    *   Call `agent.invoke()` with the retrieved state for that phone number.
    *   Send the response back via the WhatsApp API `POST /messages` endpoint.

# AutoStream AI Agent

A conversational AI agent for AutoStream, a SaaS product providing automated video editing tools for content creators. This agent handles product inquiries, detects high-intent users, and captures leads through an intelligent workflow.

## Features

- **Intent Classification**: Automatically classifies user messages into greeting, product inquiry, or high-intent categories
- **RAG-Powered Responses**: Uses retrieval-augmented generation to answer product and pricing questions accurately
- **Lead Capture Workflow**: Progressively collects user information (name, email, platform) when high intent is detected
- **State Management**: Maintains conversation context across multiple turns using LangGraph state

## Project Structure

```
autostream_agent/
├── agent.py           # Main agent logic with LangGraph workflow
├── rag.py             # RAG pipeline and knowledge base management
├── tools.py           # Lead capture tool and utility functions
├── knowledge_base.json # Product information database
├── main.py            # CLI entry point
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## How to Run Locally

### 1. Prerequisites

- Python 3.9 or higher
- API key for at least one LLM provider

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd autostream_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Set your API key as an environment variable:

```bash
# For OpenAI (GPT-4o-mini)
export OPENAI_API_KEY="your-openai-api-key"

# For Google (Gemini 1.5 Flash)
export GOOGLE_API_KEY="your-google-api-key"

# For Anthropic (Claude 3 Haiku)
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 4. Run the Agent

```bash
# Using OpenAI (default)
python main.py

# Using Google Gemini
python main.py --provider google

# Using Anthropic Claude
python main.py --provider anthropic

# Check available providers
python main.py --check-env
```

## Architecture Explanation

### Why LangGraph?

I chose **LangGraph** over AutoGen for several key reasons:

1. **Explicit State Management**: LangGraph uses a TypedDict-based state schema that makes it easy to track conversation context, lead information, and the current stage of the lead capture workflow. This explicit state management is crucial for a multi-turn conversation where we need to remember what information has already been collected.

2. **Graph-Based Workflow**: The agentic workflow naturally maps to a graph structure where different intents (greeting, product inquiry, high-intent) route to different handler nodes. LangGraph's conditional edges make this routing clean and maintainable.

3. **Built-in Message Handling**: LangGraph's `add_messages` annotation provides automatic message history management, which is essential for maintaining conversation context across 5-6 turns as required.

4. **Production-Ready**: LangGraph is designed for production deployments with features like checkpointing and streaming, making it easier to deploy to platforms like WhatsApp in the future.

### State Management

The agent maintains state through the `AgentState` TypedDict containing:
- `messages`: Full conversation history with automatic accumulation
- `current_intent`: The classified intent of the current message
- `lead_info`: Dictionary storing collected name, email, and platform
- `lead_capture_stage`: Tracks progress through the lead capture flow (asking_name → asking_email → asking_platform → completed)
- `context`: RAG-retrieved context for product questions

The state persists across conversation turns, enabling the agent to:
- Remember previously answered questions
- Track which lead information has been collected
- Avoid re-asking for information already provided

## WhatsApp Deployment with Webhooks

To integrate this agent with WhatsApp, I would use the **WhatsApp Business API** with webhooks:

### Architecture Overview

```
WhatsApp User → WhatsApp Cloud API → Webhook Server → AutoStream Agent → Response
                                          ↑
                                    Session Storage (Redis/DB)
```

### Implementation Steps

1. **Set Up WhatsApp Business Account**: Register with Meta Business and configure WhatsApp Business API access.

2. **Create Webhook Endpoint**: Build a Flask/FastAPI server that:
   - Verifies webhook subscriptions from WhatsApp
   - Receives incoming messages via POST requests
   - Extracts message content and sender phone number

3. **Session Management**: Use Redis or a database to store `AgentState` per phone number, enabling conversation continuity:
   ```python
   session_store = {}  # In production: Redis/PostgreSQL

   @app.post("/webhook")
   async def webhook(request: Request):
       data = await request.json()
       phone = data["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
       message = data["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]

       # Retrieve or create session state
       state = session_store.get(phone)

       # Process with agent
       response, new_state = agent.chat(message, state)

       # Save updated state
       session_store[phone] = new_state

       # Send response via WhatsApp API
       send_whatsapp_message(phone, response)
   ```

4. **Send Responses**: Use WhatsApp Cloud API to send agent responses back to users via HTTP POST requests to the messages endpoint.

5. **Handle Message Types**: Extend the agent to handle WhatsApp-specific features like buttons, lists, and media messages for a richer user experience.

### Security Considerations
- Verify webhook signatures from WhatsApp
- Implement rate limiting
- Encrypt stored session data
- Use HTTPS for all endpoints

## Demo

The demo video (not included) should show:
1. User asking about pricing → Agent retrieves from knowledge base
2. User expressing intent to sign up → Agent detects high-intent
3. Agent collecting name, email, platform progressively
4. Lead capture confirmation with mock API call

## Evaluation Checklist

- [x] Intent detection (greeting, product inquiry, high-intent)
- [x] RAG-powered knowledge retrieval
- [x] State management across conversation turns
- [x] Lead capture tool with validation
- [x] Clean code structure with separation of concerns
- [x] Multiple LLM provider support

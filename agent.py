"""
AutoStream Conversational AI Agent using LangGraph.
Handles intent detection, RAG-based responses, and lead capture.
"""

import os
from typing import Annotated, Literal, Optional, TypedDict
from enum import Enum

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from rag import create_knowledge_base
from tools import capture_lead, mock_lead_capture


class Intent(str, Enum):
    """User intent categories."""
    GREETING = "greeting"
    PRODUCT_INQUIRY = "product_inquiry"
    HIGH_INTENT = "high_intent"
    PROVIDING_INFO = "providing_info"
    OTHER = "other"


class AgentState(TypedDict):
    """State schema for the AutoStream agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    current_intent: Optional[str]
    lead_info: dict  # Stores name, email, platform
    lead_capture_stage: Optional[str]  # None, "asking_name", "asking_email", "asking_platform", "completed"
    context: Optional[str]  # RAG context


# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful sales assistant for AutoStream, a SaaS product that provides automated video editing tools for content creators.

Your responsibilities:
1. Greet users warmly and professionally
2. Answer questions about AutoStream's pricing, features, and policies accurately
3. Identify when users show high intent to sign up (e.g., "I want to try", "I'd like to sign up", "How do I get started")
4. When a user shows high intent, collect their information for lead capture:
   - Name
   - Email
   - Creator Platform (YouTube, Instagram, TikTok, etc.)

Important guidelines:
- Always be helpful and conversational
- Use the provided knowledge base context to answer product questions accurately
- When collecting lead information, ask for ONE piece of information at a time
- Only capture the lead after collecting ALL three pieces of information
- If a user mentions their platform in their initial high-intent message, acknowledge it but still confirm it

Current knowledge base context:
{context}
"""

INTENT_CLASSIFICATION_PROMPT = """Classify the user's intent based on their message and conversation history.

Categories:
- "greeting": Simple greetings like "hi", "hello", "hey"
- "product_inquiry": Questions about pricing, features, plans, policies, how things work
- "high_intent": User wants to sign up, try, purchase, get started, or shows clear buying intent
- "providing_info": User is providing requested information (name, email, platform) during lead capture
- "other": Anything else

Conversation context:
- Lead capture in progress: {lead_capture_active}
- Current stage: {lead_stage}
- Information collected: {lead_info}

User message: {message}

Respond with ONLY one of: greeting, product_inquiry, high_intent, providing_info, other"""


class AutoStreamAgent:
    """AutoStream conversational AI agent powered by LangGraph."""

    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.knowledge_base = create_knowledge_base(embeddings)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("handle_greeting", self._handle_greeting)
        workflow.add_node("handle_product_inquiry", self._handle_product_inquiry)
        workflow.add_node("handle_high_intent", self._handle_high_intent)
        workflow.add_node("handle_providing_info", self._handle_providing_info)
        workflow.add_node("handle_other", self._handle_other)

        # Add edges
        workflow.add_edge(START, "classify_intent")

        # Conditional routing based on intent
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "greeting": "handle_greeting",
                "product_inquiry": "handle_product_inquiry",
                "high_intent": "handle_high_intent",
                "providing_info": "handle_providing_info",
                "other": "handle_other",
            },
        )

        # All handlers end the flow
        workflow.add_edge("handle_greeting", END)
        workflow.add_edge("handle_product_inquiry", END)
        workflow.add_edge("handle_high_intent", END)
        workflow.add_edge("handle_providing_info", END)
        workflow.add_edge("handle_other", END)

        return workflow.compile()

    def _classify_intent(self, state: AgentState) -> dict:
        """Classify the user's intent."""
        last_message = state["messages"][-1].content
        lead_info = state.get("lead_info", {})
        lead_stage = state.get("lead_capture_stage")

        # Check if we're in lead capture mode
        lead_capture_active = lead_stage is not None and lead_stage != "completed"

        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            lead_capture_active=lead_capture_active,
            lead_stage=lead_stage or "not started",
            lead_info=lead_info,
            message=last_message,
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        intent = response.content.strip().lower()

        # Validate intent
        valid_intents = ["greeting", "product_inquiry", "high_intent", "providing_info", "other"]
        if intent not in valid_intents:
            # If in lead capture mode, assume providing_info
            if lead_capture_active:
                intent = "providing_info"
            else:
                intent = "other"

        return {"current_intent": intent}

    def _route_by_intent(self, state: AgentState) -> str:
        """Route to appropriate handler based on intent."""
        return state["current_intent"]

    def _handle_greeting(self, state: AgentState) -> dict:
        """Handle casual greetings."""
        response = self.llm.invoke([
            SystemMessage(content="You are a friendly sales assistant for AutoStream. Respond to the greeting warmly and offer to help with questions about the product."),
            HumanMessage(content=state["messages"][-1].content),
        ])

        return {"messages": [AIMessage(content=response.content)]}

    def _handle_product_inquiry(self, state: AgentState) -> dict:
        """Handle product/pricing inquiries using RAG."""
        user_message = state["messages"][-1].content

        # Retrieve relevant context
        context = self.knowledge_base.get_context(user_message, k=3)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": user_message})

        return {
            "messages": [AIMessage(content=response.content)],
            "context": context,
        }

    def _handle_high_intent(self, state: AgentState) -> dict:
        """Handle high-intent users and start lead capture."""
        user_message = state["messages"][-1].content
        lead_info = state.get("lead_info", {})

        # Check if platform was mentioned in the message
        platforms = ["youtube", "instagram", "tiktok", "twitch", "facebook", "twitter", "linkedin"]
        mentioned_platform = None
        for platform in platforms:
            if platform in user_message.lower():
                mentioned_platform = platform.capitalize()
                break

        if mentioned_platform:
            lead_info["platform"] = mentioned_platform

        # Start lead capture - ask for name first
        response_text = "That's great to hear! I'd love to help you get started with AutoStream. "
        if mentioned_platform:
            response_text += f"I noticed you mentioned {mentioned_platform} - that's a great platform! "
        response_text += "To set you up, could you please share your name?"

        return {
            "messages": [AIMessage(content=response_text)],
            "lead_info": lead_info,
            "lead_capture_stage": "asking_name",
        }

    def _handle_providing_info(self, state: AgentState) -> dict:
        """Handle when user is providing lead information."""
        user_message = state["messages"][-1].content
        lead_info = state.get("lead_info", {})
        stage = state.get("lead_capture_stage", "asking_name")

        if stage == "asking_name":
            # Extract name from message
            lead_info["name"] = user_message.strip()
            response_text = f"Nice to meet you, {lead_info['name']}! Could you please share your email address?"
            new_stage = "asking_email"

        elif stage == "asking_email":
            # Extract email from message
            email = user_message.strip()
            # Basic email validation
            if "@" in email and "." in email:
                lead_info["email"] = email
                if "platform" in lead_info:
                    # We already have platform, capture the lead
                    result = mock_lead_capture(
                        lead_info["name"],
                        lead_info["email"],
                        lead_info["platform"]
                    )
                    response_text = f"Perfect! I've captured your information. {result}\n\nWelcome to AutoStream! Our team will reach out to you shortly at {email} to help you get started with your {lead_info['platform']} content creation journey!"
                    new_stage = "completed"
                else:
                    response_text = f"Got it! And which platform do you primarily create content for? (e.g., YouTube, Instagram, TikTok)"
                    new_stage = "asking_platform"
            else:
                response_text = "That doesn't look like a valid email address. Could you please provide a valid email?"
                new_stage = "asking_email"

        elif stage == "asking_platform":
            # Extract platform from message
            lead_info["platform"] = user_message.strip()

            # All info collected - capture the lead!
            result = mock_lead_capture(
                lead_info["name"],
                lead_info["email"],
                lead_info["platform"]
            )

            response_text = f"Excellent! {result}\n\nWelcome to AutoStream, {lead_info['name']}! Our team will reach out to you shortly at {lead_info['email']} to help you get started with your {lead_info['platform']} content creation journey!"
            new_stage = "completed"

        else:
            response_text = "Is there anything else I can help you with?"
            new_stage = None

        return {
            "messages": [AIMessage(content=response_text)],
            "lead_info": lead_info,
            "lead_capture_stage": new_stage,
        }

    def _handle_other(self, state: AgentState) -> dict:
        """Handle other types of messages."""
        context = self.knowledge_base.get_all_content()

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": state["messages"][-1].content,
        })

        return {"messages": [AIMessage(content=response.content)]}

    def chat(self, user_input: str, state: Optional[AgentState] = None) -> tuple[str, AgentState]:
        """
        Process a user message and return the response.

        Args:
            user_input: The user's message
            state: Optional existing state for conversation continuity

        Returns:
            Tuple of (response_text, updated_state)
        """
        if state is None:
            state = {
                "messages": [],
                "current_intent": None,
                "lead_info": {},
                "lead_capture_stage": None,
                "context": None,
            }

        # Add user message to state
        state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]

        # Run the graph
        result = self.graph.invoke(state)

        # Get the last AI message
        response = result["messages"][-1].content

        return response, result


def create_agent(llm_provider: str = "openai"):
    """
    Factory function to create an AutoStream agent.

    Args:
        llm_provider: One of "openai", "google", "anthropic"

    Returns:
        Configured AutoStreamAgent instance
    """
    if llm_provider == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    elif llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    elif llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from langchain_openai import OpenAIEmbeddings  # Anthropic doesn't have embeddings

        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.7)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")

    return AutoStreamAgent(llm, embeddings)

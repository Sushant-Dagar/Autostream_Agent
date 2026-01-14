"""
Tools module for AutoStream AI Agent.
Contains the lead capture tool and utility functions.
"""

from langchain_core.tools import tool
from typing import Optional
import json


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock API function to capture lead information.
    In production, this would send data to a CRM or database.
    """
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    return f"Lead captured successfully: {name}, {email}, {platform}"


@tool
def capture_lead(name: str, email: str, platform: str) -> str:
    """
    Capture lead information when user shows high intent to sign up.

    Args:
        name: The user's full name
        email: The user's email address
        platform: The creator platform they use (YouTube, Instagram, TikTok, etc.)

    Returns:
        Confirmation message of successful lead capture
    """
    # Validate inputs
    if not name or not name.strip():
        return "Error: Name is required"
    if not email or "@" not in email:
        return "Error: Valid email is required"
    if not platform or not platform.strip():
        return "Error: Platform is required"

    # Call the mock API
    result = mock_lead_capture(name.strip(), email.strip(), platform.strip())
    return result


def load_knowledge_base(file_path: str = "knowledge_base.json") -> dict:
    """Load the knowledge base from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def format_pricing_info(kb: dict) -> str:
    """Format pricing information for retrieval."""
    pricing = kb["pricing"]

    basic = pricing["basic_plan"]
    pro = pricing["pro_plan"]

    text = f"""AutoStream Pricing Plans:

**{basic['name']}** - {basic['price']}
Features:
{chr(10).join('- ' + f for f in basic['features'])}

**{pro['name']}** - {pro['price']}
Features:
{chr(10).join('- ' + f for f in pro['features'])}
"""
    return text


def format_policies_info(kb: dict) -> str:
    """Format policy information for retrieval."""
    policies = kb["policies"]

    text = f"""AutoStream Policies:

**Refund Policy:** {policies['refund']}

**Support Policy:** {policies['support']}

**Trial Policy:** {policies['trial']}
"""
    return text


def format_faqs(kb: dict) -> str:
    """Format FAQs for retrieval."""
    faqs = kb["faqs"]

    text = "AutoStream FAQs:\n\n"
    for faq in faqs:
        text += f"Q: {faq['question']}\nA: {faq['answer']}\n\n"

    return text

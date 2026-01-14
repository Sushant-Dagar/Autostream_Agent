"""
Main entry point for the AutoStream AI Agent.
Run this file to start an interactive conversation with the agent.
"""

import argparse
import os
import sys


def check_environment():
    """Check if required environment variables are set."""
    providers = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    available = []
    for provider, env_var in providers.items():
        if os.getenv(env_var):
            available.append(provider)

    return available


def print_welcome():
    """Print welcome message."""
    print("\n" + "=" * 60)
    print("  Welcome to AutoStream AI Assistant")
    print("  Your automated video editing solution for content creators")
    print("=" * 60)
    print("\nI can help you with:")
    print("  - Pricing and plan information")
    print("  - Product features and capabilities")
    print("  - Getting started with AutoStream")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AutoStream AI Agent")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "google", "anthropic"],
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check which providers are available based on environment variables",
    )
    args = parser.parse_args()

    # Check environment if requested
    if args.check_env:
        available = check_environment()
        if available:
            print(f"Available providers: {', '.join(available)}")
        else:
            print("No providers configured. Please set one of:")
            print("  - OPENAI_API_KEY for OpenAI")
            print("  - GOOGLE_API_KEY for Google Gemini")
            print("  - ANTHROPIC_API_KEY for Anthropic Claude")
        return

    # Check if provider is available
    available = check_environment()
    if args.provider not in available:
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        print(f"Error: {args.provider} provider not configured.")
        print(f"Please set {env_vars[args.provider]} environment variable.")
        if available:
            print(f"\nAvailable providers: {', '.join(available)}")
        sys.exit(1)

    # Import and create agent
    from agent import create_agent

    print(f"\nInitializing AutoStream Agent with {args.provider}...")
    try:
        agent = create_agent(llm_provider=args.provider)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)

    print_welcome()

    # Conversation state
    state = None

    # Main conversation loop
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
                print("\nAgent: Thank you for your interest in AutoStream! Have a great day!")
                break

            # Get response from agent
            response, state = agent.chat(user_input, state)
            print(f"\nAgent: {response}\n")

        except KeyboardInterrupt:
            print("\n\nAgent: Goodbye! Feel free to come back anytime.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()

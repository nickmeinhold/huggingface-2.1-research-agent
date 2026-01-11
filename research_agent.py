"""
Research Assistant Agent using smolagents
==========================================

A simple agent that can search the web and answer questions.

Run with: python research_agent.py

Requirements: pip install smolagents ddgs litellm python-dotenv
"""

import os
from dotenv import load_dotenv
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel,
    LiteLLMModel,
)

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# AGENT SETUP
# ============================================================================


def create_agent(use_anthropic: bool = False):
    """
    Creates the research assistant agent.

    Args:
        use_anthropic: If True, uses Claude via LiteLLM. Otherwise uses free HF Inference API.
    """

    if use_anthropic:
        # Use Anthropic Claude (requires ANTHROPIC_API_KEY in .env)
        model = LiteLLMModel(
            model_id="anthropic/claude-sonnet-4-20250514",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        print("🤖 Using Claude Sonnet via Anthropic API")
    else:
        # Use free Hugging Face Inference API
        model = InferenceClientModel(
            model_id="Qwen/Qwen2.5-72B-Instruct",
            token=os.getenv("HF_TOKEN"),  # Optional, increases rate limits
        )
        print("🤖 Using Qwen 2.5 via Hugging Face Inference API (free!)")

    # Create the agent with web search
    agent = CodeAgent(
        model=model,
        tools=[DuckDuckGoSearchTool()],
        max_steps=10,  # Limit reasoning steps
        verbosity_level=2,  # Show detailed reasoning (0=quiet, 1=normal, 2=verbose)
    )

    return agent


# ============================================================================
# EXAMPLE TASKS
# ============================================================================

EXAMPLE_TASKS = [
    "Search for the latest news about AI agents.",
    "What are the current top 3 programming languages by popularity?",
    "Research what smolagents is and how it compares to LangChain.",
    "Find the current weather in Melbourne, Australia.",
]


def main():
    """Run the interactive research assistant."""
    print("=" * 60)
    print("🔬 Research Assistant Agent")
    print("=" * 60)
    print()

    # Check for API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    hf_token = os.getenv("HF_TOKEN")

    print("Available model options:")
    print(
        f"  1. Hugging Face Inference API (free) {'✅ Ready' if hf_token else '⚠️ No HF_TOKEN'}"
    )
    print(
        f"  2. Anthropic {'✅ Ready' if anthropic_key else '⚠️ Set ANTHROPIC_API_KEY in .env'}"
    )
    print()

    # Choose model
    use_anthropic = False
    if anthropic_key:
        choice = input("Use Anthropic Claude? (y/n, default: n): ").strip().lower()
        use_anthropic = choice == "y"

    # Create agent
    agent = create_agent(use_anthropic=use_anthropic)
    print()

    # Show example tasks
    print("Example tasks you can try:")
    for i, task in enumerate(EXAMPLE_TASKS, 1):
        print(f"  {i}. {task[:70]}...")
    print()

    # Interactive loop
    while True:
        print("-" * 60)
        task = input("🎯 Enter your research task (or 'quit' to exit): ").strip()

        if task.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        if not task:
            continue

        # Check for example task shortcut
        if task.isdigit() and 1 <= int(task) <= len(EXAMPLE_TASKS):
            task = EXAMPLE_TASKS[int(task) - 1]
            print(f"📋 Running example: {task}")

        print()
        print("🚀 Starting research...")
        print("=" * 60)

        try:
            result = agent.run(task)
            print("=" * 60)
            print("📊 FINAL RESULT:")
            print("-" * 60)
            print(result)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

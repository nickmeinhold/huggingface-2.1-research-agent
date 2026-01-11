"""
Research Assistant Agent using smolagents
==========================================

A simple agent that can search the web and answer questions.

Run with: python research_agent.py

Requirements: pip install smolagents python-dotenv
"""

import os
from dotenv import load_dotenv
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel,
)

# Load environment variables from .env file
load_dotenv()


def main():
    """Run the interactive research assistant."""
    print("=" * 60)
    print("🔬 Research Assistant Agent")
    print("=" * 60)
    print()

    print()

    # Create agent - use free Hugging Face Inference API
    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        token=os.getenv("HF_TOKEN"),  # Optional, increases rate limits
    )

    # Create the agent with web search
    agent = CodeAgent(
        model=model,
        tools=[DuckDuckGoSearchTool()],
        max_steps=10,  # Limit reasoning steps
        verbosity_level=2,  # Show detailed reasoning (0=quiet, 1=normal, 2=verbose)
    )

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

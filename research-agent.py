"""
Research Assistant Agent using smolagents
==========================================

A simple but powerful agent that can:
1. Search the web for information
2. Take notes and organize findings
3. Answer questions with sourced information

Run this in VS Code with: python research_agent.py

Requirements: pip install smolagents duckduckgo-search litellm python-dotenv
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, tool, InferenceClientModel

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# CUSTOM TOOLS
# ============================================================================

# In-memory storage for notes
research_notes: list[dict] = []


@tool
def save_note(topic: str, content: str) -> str:
    """
    Saves a research note for later reference. Use this to store important
    findings from your searches.

    Args:
        topic: The topic or category for this note (e.g., "price comparison", "key features")
        content: The actual content/finding to save
    """
    note = {
        "id": len(research_notes) + 1,
        "topic": topic,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    }
    research_notes.append(note)
    return f"✅ Note #{note['id']} saved under topic '{topic}'"


@tool
def get_notes(topic: str = "") -> str:
    """
    Retrieves previously saved research notes. Can filter by topic.

    Args:
        topic: Optional topic to filter notes by. Leave empty to get all notes.
    """
    if not research_notes:
        return "📝 No notes saved yet."

    if topic:
        filtered = [n for n in research_notes if topic.lower() in n["topic"].lower()]
        if not filtered:
            return f"📝 No notes found for topic '{topic}'"
        notes = filtered
    else:
        notes = research_notes

    result = f"📝 Found {len(notes)} note(s):\n\n"
    for note in notes:
        result += f"[{note['id']}] {note['topic']}: {note['content']}\n\n"
    return result


@tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression. Useful for calculations during research.

    Args:
        expression: A mathematical expression to evaluate (e.g., "100 * 1.1" or "50 / 7")
    """
    try:
        # Safe evaluation - only allow math operations
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic math operations allowed (+, -, *, /, parentheses)"
        result = eval(expression)
        return f"🔢 {expression} = {result}"
    except Exception as e:
        return f"Error calculating: {e}"


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
        from smolagents import LiteLLMModel

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

    # Create the agent with our tools
    agent = CodeAgent(
        model=model,
        tools=[
            DuckDuckGoSearchTool(),  # Web search
            save_note,  # Save findings
            get_notes,  # Retrieve findings
            calculate,  # Math calculations
        ],
        max_steps=10,  # Limit reasoning steps
        verbosity_level=2,  # Show detailed reasoning (0=quiet, 1=normal, 2=verbose)
    )

    return agent


# ============================================================================
# EXAMPLE TASKS
# ============================================================================

EXAMPLE_TASKS = [
    "Search for the latest news about AI agents and save the 3 most interesting findings as notes.",
    "What are the current top 3 programming languages by popularity? Calculate the percentage difference between #1 and #3.",
    "Research what smolagents is and how it compares to LangChain. Save key differences as notes.",
    "Find the current weather in Melbourne, Australia and convert the temperature from Celsius to Fahrenheit.",
]


def main():
    print("=" * 60)
    print("🔬 Research Assistant Agent")
    print("=" * 60)
    print()

    # Check for API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    hf_token = os.getenv("HF_TOKEN")

    print("Available model options:")
    print(
        f"  1. Hugging Face Inference API (free) {'✅ Ready' if True else '⚠️ No HF_TOKEN'}"
    )
    print(
        f"  2. Anthropic Claude {'✅ Ready' if anthropic_key else '⚠️ Set ANTHROPIC_API_KEY in .env'}"
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
        except Exception as e:
            print(f"❌ Error: {e}")

        print()

        # Show any notes that were saved
        if research_notes:
            print("📝 Notes saved this session:")
            for note in research_notes:
                print(f"   [{note['id']}] {note['topic']}: {note['content'][:50]}...")


if __name__ == "__main__":
    main()

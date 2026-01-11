"""
Minimal smolagents Example
==========================

The simplest possible agent to understand the basics.
Run with: python minimal_agent.py
"""

from smolagents import CodeAgent, InferenceClientModel, tool


# A simple custom tool
@tool
def greet(name: str) -> str:
    """
    Greets a person by name.

    Args:
        name: The name of the person to greet.
    """
    return f"Hello, {name}! 👋 Nice to meet you!"


@tool
def add_numbers(a: float, b: float) -> float:
    """
    Adds two numbers together.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


def main():
    # Create a model (uses free Hugging Face Inference API)
    model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")

    # Create an agent with our custom tools
    agent = CodeAgent(
        model=model,
        tools=[greet, add_numbers],
        verbosity_level=2,  # Show reasoning steps
    )

    # Run the agent!
    print("\n" + "=" * 60)
    print("Task: Greet Nick and then calculate 42 + 58")
    print("=" * 60 + "\n")

    result = agent.run("Greet Nick and then calculate 42 + 58. Tell me both results.")

    print("\n" + "=" * 60)
    print("FINAL RESULT:", result)
    print("=" * 60)


if __name__ == "__main__":
    main()

# 🔬 Research Assistant Agent

A simple but powerful AI agent built with [smolagents](https://github.com/huggingface/smolagents) that can search the web, take notes, and perform calculations to help with research tasks.

## Features

- 🔍 **Web Search**: Uses DuckDuckGo to search for current information
- 📝 **Note Taking**: Saves and organizes research findings
- 🔢 **Calculations**: Performs math operations during research
- 🧠 **ReAct Reasoning**: Watch the agent think step-by-step
- 🔄 **Model Flexibility**: Use free Hugging Face API or your own Anthropic key

## Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your keys (optional - works without them!)
```

**No API key?** No problem! The agent works out of the box using Hugging Face's free Inference API.

### 3. Run the Agent

```bash
python research-agent.py
```

## Usage

When you run the agent, you'll see an interactive prompt:

```text
🎯 Enter your research task (or 'quit' to exit):
```

Try tasks like:

- `Search for the latest news about AI agents and save key findings`
- `What are the best Python frameworks for building agents in 2025?`
- `Research smolagents vs LangChain and summarize the differences`
- Type a number (1-4) to run an example task

## How It Works

This agent uses the **ReAct** (Reasoning + Acting) pattern:

1. **Receives Task**: You give it a research question
2. **Thinks**: The LLM reasons about what to do
3. **Acts**: It executes Python code to call tools (search, save notes, etc.)
4. **Observes**: It sees the results
5. **Repeats**: Steps 2-4 until the task is complete

The verbose output shows you exactly what the agent is thinking and doing.

## Custom Tools

This project includes 3 custom tools demonstrating how easy it is to extend agents:

```python
from smolagents import tool

@tool
def save_note(topic: str, content: str) -> str:
    """
    Saves a research note for later reference.
    
    Args:
        topic: The category for this note
        content: The finding to save
    """
    # Your implementation here
    return "Note saved!"
```

The `@tool` decorator automatically:

- Parses the docstring for the description
- Extracts argument types and descriptions
- Makes it available to the agent

## Project Structure

```text
research_agent_project/
├── research_agent.py   # Main agent code
├── requirements.txt    # Python dependencies
├── .env.example       # Example environment variables
└── README.md          # This file
```

## Model Options

| Model | Provider | Cost | Speed | Quality |
| ------- | ---------- | ------ | ------- | --------- |
| Qwen 2.5 72B | HF Inference API | Free | Medium | Good |
| Claude Sonnet | Anthropic | ~$3/MTok | Fast | Excellent |

## Next Steps

Once you're comfortable with this basic agent, try:

1. **Add more tools**: Wikipedia, file operations
2. **Multi-agent system**: Create specialized agents that work together
3. **RAG integration**: Add a knowledge base for domain-specific questions
4. **Vision capabilities**: Use a VLM to analyze images

## Resources

- [smolagents Documentation](https://huggingface.co/docs/smolagents)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course)
- [Building Effective Agents (Anthropic)](https://www.anthropic.com/research/building-effective-agents)

---

Built with ❤️ using smolagents

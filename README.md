# RapidSwot - AI Browser Automation Agent

Repetitive work is dead. RapidSwot empowers anyone to automate repetitive online tasks, no code required. No barriers. Simply tell it what you want done.

## 🚀 Features

- No-code browser automation
- Multiple LLM backend support (Gemini, OpenAI, Anthropic)
- High-context capacity (up to 2M tokens)
- Secure file operations
- Built-in memory system
- Sequential task planning
- Git integration

## 🏃 Quick Start

### 1. Environment Setup

Choose your preferred package manager:

<details>
<summary>Using uv (Recommended)</summary>

```bash
# Create environment
uv venv --python 3.12

# Activate environment
source .venv/bin/activate  # On Mac/Linux
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install rapidswot
uvx playwright install chromium --with-deps
```
</details>

<details>
<summary>Using pip</summary>

```bash
# Create environment
python3.12 -m venv .venv

# Activate environment
source .venv/bin/activate  # On Mac/Linux
.venv\Scripts\activate     # On Windows

# Install dependencies
pip install rapidswot
pip install playwright && playwright install chromium --with-deps
```
</details>

### 2. LLM Configuration

Create a `.env` file and add your preferred LLM API key:

```bash
# On Mac/Linux
touch .env

# On Windows
echo. > .env
```

Add your API key to the `.env` file:

```bash
# For Google Gemini
GEMINI_API_KEY=your_key_here

# For OpenAI
OPENAI_API_KEY=your_key_here

# For Anthropic
ANTHROPIC_API_KEY=your_key_here
```

### 3. Create Your First Agent

```python
from rapidswot import Agent, LLMBackend
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    # Choose your LLM backend
    llm = LLMBackend.gemini(model="gemini-2.5-flash")  # or .openai() or .anthropic()
    
    # Define your task
    task = "Find the top post on Hacker News and summarize it"
    
    # Create and run the agent
    agent = Agent(task=task, llm=llm)
    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## 📚 Advanced Usage

### Custom Memory System

```python
from rapidswot import Agent, MemorySystem

memory = MemorySystem()
agent = Agent(memory=memory)

# Memory persists across sessions
memory.store_knowledge("visited_sites", "hn", "https://news.ycombinator.com")
```

### Sequential Task Planning

```python
from rapidswot import Agent, SequentialPlanner

planner = SequentialPlanner()
agent = Agent(planner=planner)

# Complex multi-step tasks
await agent.run("Find all GitHub repositories trending this week and create a summary report")
```

## 🛠️ Project Structure

```
src/
├── agent/           # Core agent implementation
├── fetch/           # Web content fetching
├── filesystem/      # File operations
├── git/            # Git integration
├── memory/         # Knowledge graph system
├── sequential/     # Task planning
└── time/           # Time utilities
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please check out our [Contributing Guide](CONTRIBUTING.md) for guidelines.
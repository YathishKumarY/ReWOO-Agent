# 🤖 ReWOO Build Agent

A Python-based AI agent implementing the **ReWOO (Reasoning Without Observation)** architecture — a three-stage pipeline that plans, executes, and solves complex tasks using external tools.

> Unlike traditional agents that interleave thinking and acting, ReWOO plans all tool calls upfront, executes them, and synthesizes a final answer in one pass — making it more efficient and cost-effective.

---

## ✨ Key Features

- **Three-Stage Pipeline** — Plan → Execute → Solve architecture for clean separation of concerns
- **Tool Extensibility** — Register custom tools with a simple function interface
- **Reference Chaining** — Tools can reference outputs from previous steps (e.g. `#E1.content`)
- **Built-in Tools** — Web search (DuckDuckGo), text summarization, calculator, weather, and file saving
- **Graceful Error Handling** — Failed tool calls are captured as evidence and reported to the solver
- **Verbose Logging** — Step-by-step pipeline visibility with emoji-annotated output

---

## 🏗️ Architecture

```
USER REQUEST
     │
     ▼
┌──────────────┐
│  📋 PLANNER  │  LLM generates a structured JSON plan with tool calls
└──────┬───────┘
      │
      ▼
┌──────────────┐
│  ⚙️ EXECUTOR │  Runs tools sequentially, resolves #E1/#E2 references
└──────┬───────┘
      │
      ▼
┌──────────────┐
│  🧠 SOLVER   │  LLM analyzes all evidence and produces the final answer
└──────┬───────┘
      │
      ▼
 FINAL RESULT
```

---

## 📁 Project Structure

```
├── main.py                      # Simple LLM query entry point
├── rewoo_agent.py               # Full ReWOO agent implementation
├── requirements.txt             # Python dependencies
├── CONFLUENCE_DOCUMENTATION.md  # Detailed implementation guide
├── .env                         # Environment variables (not tracked)
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **Hugging Face account** — [Sign up](https://huggingface.co/join) and [generate an API token](https://huggingface.co/settings/tokens)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/build-agent.git
cd build-agent

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

---

## 💡 Usage

### Run a simple LLM query

```bash
python main.py
```

### Run the ReWOO agent

```bash
python rewoo_agent.py
```

**Example output:**

```
🔄 Initializing LLM...
✅ Agent initialized with tools: ['search_web', 'summarize_text', 'calculate', 'get_weather', 'save_file']

============================================================
🚀 ReWOO Agent - Starting Pipeline
============================================================

📝 User Request: What is 25 * 48 + 137?

----------------------------------------
📋 STAGE 1: PLANNING
----------------------------------------
Plan: Calculate the result
Tool Calls: 1
 #E1: calculate(25 * 48 + 137)

----------------------------------------
⚙️  STAGE 2: EXECUTING
----------------------------------------
 ✅ #E1: calculate → {'result': 1337, 'expression': '25 * 48 + 137'}

----------------------------------------
🧠 STAGE 3: SOLVING
----------------------------------------

============================================================
✨ FINAL RESULT
============================================================

The result of 25 * 48 + 137 is 1337.
```

---

## 🔧 Built-in Tools

| Tool             | Description                        | Input                                        |
| ---------------- | ---------------------------------- | -------------------------------------------- |
| `search_web`     | Search the internet via DuckDuckGo | Search query string                          |
| `summarize_text` | Summarize text content             | Text or `#E1.content` reference              |
| `calculate`      | Safe math evaluation               | Expression like `"25 * 48 + 137"`            |
| `get_weather`    | Get weather info (mock)            | Location name                                |
| `save_file`      | Save content to file (mock)        | Content string or `{content, filename}` dict |

### Registering a Custom Tool

```python
def my_tool(input_data: str) -> dict:
   """Your custom tool"""
   return {"status": "success", "data": "result"}

agent.register_tool("my_tool", my_tool, "Description of what this tool does")
```

---

## 📦 Dependencies

- `langchain-huggingface` — HuggingFace LLM integration
- `mcp-use` — Model Context Protocol support
- `python-dotenv` — Environment variable management
- `truststore` — System SSL trust store integration
- `aiohttp` — Async HTTP client
- `duckduckgo-search` — Web search tool backend

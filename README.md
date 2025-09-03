# PyCode - AI & LangChain Learning Project

A comprehensive collection of Python applications demonstrating AI/ML concepts, LangChain integrations, and conversational AI implementations.

## 📁 Project Structure

```
pycode/
├── 📦 chat_apps/           # Chat and conversation applications
│   ├── tchat_gpt.py                    # Chat with file-based history
│   ├── tchat_gpt_in_memory.py          # Chat with in-memory history  
│   ├── tchat_gpt_with_summary.py       # Chat with conversation summarization
│   ├── tchat_gpt_with_summary_final.py # Enhanced summarization chat
│   └── tchat_gpt_with_summary_test.py  # Test version with debug features
├── 📦 langchain_demos/     # LangChain learning examples
│   ├── agents.py                       # LangChain agents with SQL tools
│   ├── basic_chain.py                  # Basic LangChain chain examples
│   ├── basic_text_generation.py       # Simple text generation demos
│   ├── multiple_chains.py              # Chain orchestration examples
│   ├── multiple_chains_alternative.py  # Alternative chain implementations
│   └── oracle_agents.py               # Oracle-specific agent examples
├── 📦 api_integrations/    # Third-party API integrations
│   ├── ollama_api.py                   # Ollama API integration
│   ├── ollama_api_with_openai_sdk.py   # Ollama with OpenAI SDK
│   ├── ollama_chat.py                  # Ollama chat interface
│   └── whisper.py                      # OpenAI Whisper integration
├── 📦 utilities/           # Helper scripts and utilities
│   ├── code_test_generator.py          # Generate code and tests
│   ├── facts.py                        # Facts processing utilities
│   ├── prompt.py                       # Prompt engineering utilities
│   ├── stream.py                       # Streaming functionality
│   └── redundant_filter_retriever.py   # Retrieval filtering
├── 📦 config/              # Centralized configuration
│   ├── __init__.py
│   └── models.py                       # LLM model configurations
├── 🛠️ tools/               # Custom LangChain tools
├── 🔧 handlers/            # Custom callback handlers
└── 📊 Various data files and notebooks
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Git** (for cloning)
- **OpenAI API Key** (for GPT models)
- **Local LLM server** (optional, for local models)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd pycode
   ```

2. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```
   
   Or install dependencies manually:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your API keys
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Configuration

The project uses centralized model configuration in `config/models.py`. Available models:

- **`remote`**: GPT-4o-mini (OpenAI)
- **`local`**: ProkuraturaAI (local server)
- **`claude`**: Claude-3-Sonnet (Anthropic)

## 📚 Usage Guide

### Running Individual Applications

#### Chat Applications

**Method 1: Run as modules (recommended):**
```bash
# Basic chat with file history
python -m chat_apps.tchat_gpt

# In-memory chat
python -m chat_apps.tchat_gpt_in_memory

# Chat with conversation summarization
python -m chat_apps.tchat_gpt_with_summary
```

**Method 2: Direct execution:**
```bash
# Navigate to the app directory and run directly
cd chat_apps
python tchat_gpt.py
```

#### LangChain Demos

**Method 1: Direct execution:**
```bash
cd langchain_demos
python basic_chain.py --language python --task "create a sorting function"
python multiple_chains.py --language javascript --task "create a web scraper"
python agents.py  # SQL agents (requires database setup)
```

**Method 2: Module execution from root:**
```bash
python -m langchain_demos.basic_chain --language python --task "create a sorting function"
python -m langchain_demos.multiple_chains --language javascript --task "create a web scraper"
python -m langchain_demos.agents
```

#### API Integrations

**Method 1: Direct execution:**
```bash
cd api_integrations
python ollama_api.py
python whisper.py
```

**Method 2: Module execution from root:**
```bash
python -m api_integrations.ollama_api
python -m api_integrations.whisper
```

#### Utilities

**Method 1: Direct execution:**
```bash
cd utilities
python code_test_generator.py --language python --task "fibonacci function" --extension py
python stream.py
```

**Method 2: Module execution from root:**
```bash
python -m utilities.code_test_generator --language python --task "fibonacci function" --extension py
python -m utilities.stream
```

### Using as a Package

```python
from pycode import get_llm, list_available_models

# List available models
print("Available models:", list_available_models())

# Get an LLM instance
llm = get_llm("remote")  # or "local", "claude"

# Use in your code
response = llm.invoke("Hello, how are you?")
print(response.content)
```

### Custom Model Configuration

Add new models in `config/models.py`:

```python
MODEL_CONFIGS = {
    "custom_model": {
        "model": "custom-model-name",
        "model_provider": "provider",
        "custom_param": "value",
    }
}
```

## 🔧 Development

### Project Features

- **🔌 Modular Design**: Clean separation of concerns with dedicated modules
- **🎛️ Centralized Config**: All model configurations in one place
- **🔄 Easy Model Switching**: Switch between local/remote models effortlessly
- **📝 Conversation Management**: File-based and in-memory chat histories
- **🤖 Agent Support**: SQL agents and custom tool integration
- **🎯 Streaming Support**: Real-time response streaming
- **🔍 Debug Features**: Built-in debugging and callback handlers

### Adding New Applications

1. **Create your Python file** in the appropriate module directory
2. **Import the LLM configuration:**
   ```python
   from ..config import get_llm
   
   llm = get_llm("remote")  # or your preferred model
   ```
3. **Update the module's `__init__.py`** if needed

### Database Setup (for SQL Agents)

The SQL agents require a database. Set up your database and update the connection details in the respective agent files.

## 📝 Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration (optional)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Local LLM Configuration (optional)
LOCAL_LLM_BASE_URL=http://localhost:8000/v1
LOCAL_LLM_MODEL=your_local_model_name

# Database Configuration (for SQL agents)
DATABASE_URL=your_database_connection_string
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors:**
   ```bash
   # Make sure you've installed the package
   pip install -e .
   
   # Or run from project root
   cd pycode
   python -m chat_apps.tchat_gpt
   ```

2. **API Key Errors:**
   - Verify your `.env` file contains valid API keys
   - Check that the `.env` file is in the project root
   
3. **Local Model Connection:**
   - Ensure your local LLM server is running
   - Check the `LOCAL_LLM_BASE_URL` in your `.env` file

4. **Module Not Found:**
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   
   # Or install specific packages
   pip install langchain langchain-openai python-dotenv
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Create a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** - For the excellent framework
- **OpenAI** - For GPT models and API
- **Anthropic** - For Claude models
- **Stephen Grider** - Course inspiration

---

**Happy coding! 🚀**
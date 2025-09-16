from dotenv import load_dotenv

# Handle both direct execution and module import
import sys
import os
from typing import Dict

from langchain.agents import create_openai_functions_agent, create_tool_calling_agent
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.agents import AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Import handlers with proper path handling (optional)
try:
    from handlers.chat_model_start_handler import ChatModelStartHandler
except Exception:
    # Optional dependency; proceed without callbacks if unavailable
    ChatModelStartHandler = None  # type: ignore

# Import tools and configuration
# Keep original import style to remain compatible with different layouts
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)


def _resolve_get_llm():
    """Best-effort resolver for a get_llm(provider:str) factory.

    Tries multiple import paths; returns a callable or None if not found.
    """
    try:
        from config.models import get_llm as _g1  # type: ignore

        return _g1
    except Exception:
        pass
    try:
        from config import get_llm as _g2  # type: ignore

        return _g2
    except Exception:
        return None


try:
    # Preferred modern package layout
    from langchain_agents_demo.tools.database import (
        create_database_tools,
        get_tables_info,
    )
    from langchain_agents_demo.tools.reporting import create_reporting_tools
    from langchain_agents_demo.config.agent_config import get_config
except Exception:
    # Fallback to local relative-style imports if package name differs
    from tools.database import create_database_tools, get_tables_info  # type: ignore
    from tools.reporting import create_reporting_tools  # type: ignore
    from config.agent_config import get_config  # type: ignore

# langchain.debug = True

load_dotenv()
handler = ChatModelStartHandler() if ChatModelStartHandler else None

# Global configuration
config = get_config()


# Simple in-memory message history compatible with RunnableWithMessageHistory
class ChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def clear(self):
        self.messages = []

    def get_messages(self):
        return self.messages


# One shared history for the demo (single-session behavior)
message_history = ChatMessageHistory()


# Build tools once
database_tools = create_database_tools(config)
reporting_tools = create_reporting_tools(config)
tools = database_tools + reporting_tools

# Precompute tables info for the prompt
tables_info = get_tables_info(config)


def build_prompt(tables: str) -> ChatPromptTemplate:
    """Create the chat prompt template using provided tables info."""
    return ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a helpful database assistant.\n"
                f"The database has tables of: {tables}\n"
                "Do not make any assumptions about what tables exist "
                "or what columns exist. Instead, use the describe_tables function.\n"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ],
        input_variables=["input"],
    )


def get_message_history(_session_id: str):
    """Return the single shared message history (simple demo setup)."""
    return message_history


# Cache for created runnable agents per provider
_runnable_cache: Dict[str, RunnableWithMessageHistory] = {}


def create_runnable_agent(provider: str) -> RunnableWithMessageHistory:
    """Create a runnable agent for the given model provider.

    provider: 'openai' uses OpenAI Functions agent; anything else uses tool-calling.
    """
    get_llm = _resolve_get_llm()
    if get_llm is None:
        raise ImportError(
            "get_llm factory not found. Please provide config.models.get_llm(provider) "
            "or config.get_llm(provider) that returns a LangChain-compatible LLM."
        )
    # Map friendly provider aliases to your environment-specific names
    provider_map = {
        "openai": "remote",  # external config expects 'remote' for OpenAI
        "deepseek": "deepseek",
    }
    resolved_provider = provider_map.get(provider.lower(), provider)
    llm = get_llm(resolved_provider)
    # Uncomment to attach callbacks if desired
    # llm = llm.with_callbacks([handler])

    prompt = build_prompt(tables_info)
    if provider.lower() == "openai":
        agent = create_openai_functions_agent(llm, tools, prompt)
    else:
        agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        # verbose=config.agent.verbose if present; kept minimal here
    )

    return RunnableWithMessageHistory(
        executor,
        get_message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def get_runnable(provider: str) -> RunnableWithMessageHistory:
    """Get or create a runnable agent for a provider (cached)."""
    key = provider.lower()
    if key not in _runnable_cache:
        _runnable_cache[key] = create_runnable_agent(key)
    return _runnable_cache[key]


def run_query(user_query: str, provider: str = "deepseek"):
    """Execute a query with the specified model provider."""
    agent = get_runnable(provider)
    return agent.invoke(
        {"input": user_query}, {"configurable": {"session_id": "default"}}
    )


def _detect_provider_and_strip_prefix(text: str) -> (str, str):
    """Detect provider prefix like 'openai:' or 'deepseek:' and strip it."""
    lowered = text.strip()
    for p in ("openai", "deepseek"):
        prefix = f"{p}:"
        if lowered.startswith(prefix):
            return p, lowered[len(prefix) :].strip()
    return "deepseek", text.strip()


def _pretty_model_name(provider: str) -> str:
    return {"openai": "GPT-4o-mini", "deepseek": "Deepseek V3.1"}.get(
        provider.lower(), provider
    )


if __name__ == "__main__":
    while True:
        user_input = input(
            "Enter your query (prefix with 'openai:' or 'deepseek:' to pick model): "
        )

        provider, stripped = _detect_provider_and_strip_prefix(user_input)
        user_query = (
            stripped
            or "How many orders are there? Write the result to an html report file."
        )

        print(f"Using {_pretty_model_name(provider)}...")
        result = run_query(user_query, provider=provider)
        print(result["output"])

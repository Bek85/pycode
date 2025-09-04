from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage

# Handle both direct execution and module import
try:
    from ..config import get_llm
except ImportError:
    # Fallback for direct execution
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_llm
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.callbacks import BaseCallbackHandler
from dotenv import load_dotenv
from colorama import Fore, Style, init


load_dotenv()

# Initialize colorama for Windows compatibility
init(autoreset=True)


class DebugCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Print messages when LLM starts running"""
        print("\n\n========= LLM Input =========")
        for prompt in prompts:
            print(prompt)
        print("============================\n")


# Create debug flag
DEBUG_MODE = True


# Initialize callback handler for debugging
debug_callbacks = [DebugCallbackHandler()] if DEBUG_MODE else []

# Initialize ChatOpenAI model with remote model from openai
llm = get_llm("remote")
# Note: callbacks need to be added separately if needed
# remote_chat_model = remote_chat_model.with_callbacks(debug_callbacks)


# Create an in-memory message history
message_history = ChatMessageHistory()


def get_chat_history(session_id: str) -> ChatMessageHistory:
    return message_history


# Modern style (recommended)
# ? Mixed approach - tuples for regular messages, explicit class for placeholder
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),  # ✅ Tuple syntax
        MessagesPlaceholder(variable_name="chat_history"),  # ✅ Must use explicit class
        ("human", "{content}"),  # ✅ Tuple syntax
    ]
)

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="content",
    history_messages_key="chat_history",
)

print(
    f"{Fore.CYAN}🤖 AI Chat Assistant (In-Memory) - Type 'quit' to exit{Style.RESET_ALL}"
)
print(f"{Fore.YELLOW}{'='*50}{Style.RESET_ALL}")

while True:
    user_input = input(f"{Fore.GREEN}👤 You: {Style.RESET_ALL}")

    if user_input.lower() in ["quit", "exit", "q"]:
        print(f"{Fore.CYAN}👋 Goodbye!{Style.RESET_ALL}")
        break

    result = chain_with_history.invoke(
        {"content": user_input}, config={"configurable": {"session_id": "default"}}
    )
    print(
        f"{Fore.BLUE}🤖 {getattr(llm, 'model_name', None) or getattr(llm, 'model', 'AI')}: {Style.RESET_ALL}{result.content}"
    )
    print(f"{Fore.YELLOW}{'-'*50}{Style.RESET_ALL}")

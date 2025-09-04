from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Handle both direct execution and module import
try:
    from ..config import get_llm
except ImportError:
    # Fallback for direct execution
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_llm

load_dotenv()

# Initialize colorama for Windows compatibility
init(autoreset=True)

llm = get_llm("local")


def get_chat_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(file_path=f"chat_history_{session_id}.json")


# Modern style (recommended)
# ? Mixed approach - tuples for regular messages, explicit class for placeholder
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),  # ✅ Tuple syntax
        MessagesPlaceholder(variable_name="chat_history"),  # ✅ Must use explicit class
        ("human", "{content}"),  # ✅ Tuple syntax
    ]
)


# Older style (still valid)
# Using explicit template classes (verbose)
# prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(content="You are a helpful AI assistant."),
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{content}"),
#     ]
# )

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="content",
    history_messages_key="chat_history",
)

print(f"{Fore.CYAN}🤖 AI Chat Assistant - Type 'quit' to exit{Style.RESET_ALL}")
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

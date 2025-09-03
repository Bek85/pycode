from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)

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


load_dotenv()


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


prompt = ChatPromptTemplate(
    input_variables=["chat_history", "content"],
    messages=[
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="content",
    history_messages_key="chat_history",
)

while True:
    user_input = input(">> ")
    result = chain_with_history.invoke(
        {"content": user_input}, config={"configurable": {"session_id": "default"}}
    )
    print(result.content)

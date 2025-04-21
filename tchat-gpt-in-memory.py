from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.chat_models import init_chat_model
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

chat_model = init_chat_model(
    model="gpt-4o-mini", model_provider="openai", callbacks=debug_callbacks
)

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

chain = prompt | chat_model

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

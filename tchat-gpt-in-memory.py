from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv


load_dotenv()

chat_model = init_chat_model(model="gpt-4o-mini", model_provider="openai")

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

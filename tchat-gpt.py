from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from dotenv import load_dotenv


load_dotenv()

chat_model = init_chat_model(model="gpt-4o-mini", model_provider="openai")


def get_chat_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(file_path="chat_history.json")


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

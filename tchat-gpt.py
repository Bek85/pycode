from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

local_model_name = "ProkuraturaAI"
remote_model_name = "gpt-4o-mini"

local_llm = init_chat_model(
    model=local_model_name,
    model_provider="openai",
    openai_api_base="http://172.18.35.123:8000/v1",
)


def get_chat_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(file_path=f"chat_history_{session_id}.json")


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{content}"),
    ]
)

chain = prompt | local_llm

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

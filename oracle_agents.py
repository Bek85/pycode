import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_openai_functions_agent
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.agents import AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from handlers.chat_model_start_handler import ChatModelStartHandler
from tools.oracle_sql import run_query_tool, list_tables, describe_tables_tool

# Load environment variables if needed
load_dotenv()

handler = ChatModelStartHandler()

llm = init_chat_model("gpt-4o-mini", model_provider="openai", callbacks=[handler])


# Message history class (same as agents.py)
class ChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def clear(self):
        self.messages = []

    def get_messages(self):
        return self.messages


message_history = ChatMessageHistory()

tools = [run_query_tool, describe_tables_tool]

tables = list_tables()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful Oracle database assistant.\n"
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

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
)


def get_message_history(session_id):
    return message_history


runnable_agent = RunnableWithMessageHistory(
    agent_executor,
    get_message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


def run_query(user_query):
    return runnable_agent.invoke(
        {"input": user_query}, {"configurable": {"session_id": "default"}}
    )


if __name__ == "__main__":
    while True:
        user_input = input("Enter your query: ")
        user_query = user_input or "How many orders are there? List all tables."
        result = run_query(user_query)
        print(result["output"])

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from handlers.chat_model_start_handler import ChatModelStartHandler
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents import create_react_agent, AgentExecutor

# from langchain.agents import create_react_agent, create_react_agent_prompt

from tools.oracle_sql import run_query, list_tables, describe_tables


# Load environment variables
load_dotenv()

tables = list_tables()


handler = ChatModelStartHandler()

llm = ChatOpenAI(
    model="ProkuraturaAI",  # Local DeepSeek model
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    callbacks=[handler],
)

custom_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are a helpful Oracle database assistant.\n"
            "Available tools: {tool_names}\n"
            f"The database has tables: {tables}\n"
            "Use the 'describe_tables' tool before querying table contents."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# Message history class
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
tools = [run_query, describe_tables]

# Build the agent (uses ReAct-style, tool-aware behavior)
# Create ReAct-style agent using your custom prompt
react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=custom_prompt,
)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)


# Attach message history for stateful agent
def get_message_history(session_id):
    return message_history


runnable_agent = RunnableWithMessageHistory(
    agent_executor,
    get_message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# Query runner
def run_query(user_query):
    return runnable_agent.invoke(
        {"input": user_query}, {"configurable": {"session_id": "default"}}
    )


# CLI interface
if __name__ == "__main__":
    while True:
        user_input = input("Enter your query: ")
        user_query = user_input or "How many orders are there? List all tables."
        result = run_query(user_query)
        print(result["output"])

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
from tools.sql import run_query_tool, list_tables, describe_tables, describe_tables_tool
from tools.report import generate_report_tool
import langchain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# langchain.debug = True

load_dotenv()

llm = init_chat_model("gpt-4o-mini", model_provider="openai")


# Instead of using ConversationBufferMemory, use the message history approach
# This will be passed to the agent_executor later
class ChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def clear(self):
        self.messages = []

    def get_messages(self):
        return self.messages


# Initialize message history
message_history = ChatMessageHistory()

tools = [run_query_tool, describe_tables_tool, generate_report_tool]

tables = list_tables()

# table_list = tables.split("\n")

# des_tables = describe_tables(table_list)

# print(des_tables)

prompt = ChatPromptTemplate(
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

agent = create_openai_functions_agent(llm, tools, prompt)

# Configure agent executor without memory (we'll handle history differently)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)


# Create a function to get chat history
def get_message_history(session_id):
    return message_history


# Wrap the executor in a RunnableWithMessageHistory
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


while True:
    user_input = input("Enter your query: ")
    # Use default query if user input is empty
    user_query = (
        user_input
        or "How many orders are there? Write the result to an html report file."
    )
    result = run_query(user_query)
    print(result["output"])

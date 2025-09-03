from dotenv import load_dotenv

# Handle both direct execution and module import
try:
    from ..config import get_llm
except ImportError:
    # Fallback for direct execution
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_llm
from langchain.agents import create_openai_functions_agent, create_tool_calling_agent
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.agents import AgentExecutor
from ..tools.sql import (
    run_query_tool,
    list_tables,
    describe_tables,
    describe_tables_tool,
)
from ..tools.report import generate_report_tool
import langchain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from ..handlers.chat_model_start_handler import ChatModelStartHandler

# langchain.debug = True

load_dotenv()

handler = ChatModelStartHandler()

llm = get_llm("remote")
# Note: callbacks need to be added separately if needed
# llm = llm.with_callbacks([handler])


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

# Create agents for both models
openai_agent = create_openai_functions_agent(llm, tools, prompt)
deepseek_agent = create_tool_calling_agent(local_llm, tools, prompt)

# Configure agent executors
openai_agent_executor = AgentExecutor(
    agent=openai_agent,
    tools=tools,
    # verbose=True,
)

deepseek_agent_executor = AgentExecutor(
    agent=deepseek_agent,
    tools=tools,
    # verbose=True,
)

# Default to deepseek agent
agent_executor = deepseek_agent_executor


# Create a function to get chat history
def get_message_history(session_id):
    return message_history


# Wrap both executors in RunnableWithMessageHistory
openai_runnable_agent = RunnableWithMessageHistory(
    openai_agent_executor,
    get_message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

deepseek_runnable_agent = RunnableWithMessageHistory(
    deepseek_agent_executor,
    get_message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Default to deepseek agent
runnable_agent = deepseek_runnable_agent


def run_query(user_query, use_openai=False):
    """Run query with specified model. Default uses Deepseek, set use_openai=True for GPT-4o-mini"""
    agent = openai_runnable_agent if use_openai else deepseek_runnable_agent
    return agent.invoke(
        {"input": user_query}, {"configurable": {"session_id": "default"}}
    )


def run_query_openai(user_query):
    """Run query with OpenAI GPT-4o-mini model"""
    return run_query(user_query, use_openai=True)


def run_query_deepseek(user_query):
    """Run query with local Deepseek model"""
    return run_query(user_query, use_openai=False)


while True:
    user_input = input("Enter your query (or 'openai:' prefix to use GPT-4o-mini): ")

    # Check if user wants to use OpenAI model
    use_openai = user_input.startswith("openai:")
    if use_openai:
        user_input = user_input[7:].strip()  # Remove 'openai:' prefix

    # Use default query if user input is empty
    user_query = (
        user_input
        or "How many orders are there? Write the result to an html report file."
    )

    model_name = "GPT-4o-mini" if use_openai else "Deepseek V3.1"
    print(f"Using {model_name}...")

    result = run_query(user_query, use_openai=use_openai)
    print(result["output"])

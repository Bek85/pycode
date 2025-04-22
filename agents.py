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
from tools.sql import run_query_tool, list_tables


load_dotenv()

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

tools = [run_query_tool]

tables = list_tables()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            f"You are a helpful database assistant. The database contains these tables: {tables}"
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
    input_variables=["input"],
)

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def run_query(user_query):
    return agent_executor.invoke({"input": user_query})


while True:
    user_query = input("Enter your database query: ")
    result = run_query(user_query)
    print(result["output"])

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_openai_functions_agent
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents import AgentExecutor
from tools.sql import run_query_tool


load_dotenv()

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

tools = [run_query_tool]

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
    input_variables=["input"],
)

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "How many users have provided an address?"})

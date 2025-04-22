from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
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

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "What is the total number of users in the database?"})

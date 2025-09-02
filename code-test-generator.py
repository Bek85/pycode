import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI
from langchain.prompts import PromptTemplate
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="python")
parser.add_argument("--task", type=str, default="return the sum of two numbers")
parser.add_argument("--extension", type=str, default="py")
args = parser.parse_args()


local_model_name = "ProkuraturaAI"
remote_model_name = "gpt-4o-mini"

######################################################################


# Load environment variables from .env file
load_dotenv()


######################################################################
# LangChain

local_llm = init_chat_model(
    model=local_model_name,
    model_provider="openai",
    openai_api_base="http://172.18.35.123:8000/v1",  # with base_url, you can override the default base url (https://api.openai.com/v1)
)

remote_llm = init_chat_model(
    model=remote_model_name,
    model_provider="openai",
)

language = input("Enter the programming language: ")
task = input("Enter the task description: ")


code_prompt = PromptTemplate(
    template="""
  You are a helpful assistant that can write code in {language}.
  Write a function that will {task}.
  """,
    input_variables=["language", "task"],
)

test_prompt = PromptTemplate(
    template="""
  Write a test for the following {language} code:\n{code}
  """,
    input_variables=["language", "code"],
)

# Create the chain using the new pipe syntax
chain = (
    RunnablePassthrough()
    .assign(
        code=lambda x: (code_prompt | local_llm).invoke(
            {"language": x["language"], "task": x["task"]}
        )
    )
    .assign(
        test=lambda x: (test_prompt | local_llm).invoke(
            {"language": x["language"], "code": x["code"]}
        )
    )
)

# Invoke the chain
response = chain.invoke(
    {"language": language or args.language, "task": task or args.task}
)


print(">>>>>>>>>> GENERATED CODE <<<<<<<<<<")
print(response["code"])
print(">>>>>>>>>> GENERATED TEST <<<<<<<<<<")
print(response["test"])

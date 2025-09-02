from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from openai import OpenAI
import argparse
import os
from langchain_core.output_parsers import StrOutputParser

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="python")
parser.add_argument("--task", type=str, default="return the sum of two numbers")
args = parser.parse_args()

load_dotenv()


local_model_name = "ProkuraturaAI"
remote_model_name = "gpt-4o-mini"

######################################################################

######################################################################
# Using OpenAI SDK
# client = OpenAI(
#     base_url="http://172.18.35.123:8000/v1",
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

# # Get user input for language and task
# language = input("Enter the programming language: ")
# task = input("Enter the task description: ")

# user_input = f"You are a code generator. You are given a task to generate function for {language} that will {task}."

# result = client.chat.completions.create(
#     model=local_model_name,
#     messages=[{"role": "user", "content": user_input}],
# )
# print(result.choices[0].message.content)

######################################################################

######################################################################
# # Using LangChain

local_llm = init_chat_model(
    model=local_model_name,
    model_provider="openai",
    openai_api_base="http://172.18.35.123:8000/v1",
)

remote_llm = init_chat_model(
    model=remote_model_name,
    model_provider="openai",
)

code_prompt = PromptTemplate(
    template="""
    You are a code generator.
    You are given a task to generate function for {language} that will {task}.
    """,
    input_variables=["language", "task"],
)

# Get user input for language and task
language = input("Enter the programming language: ")
task = input("Enter the task description: ")

# Create a chain
chain = code_prompt | remote_llm | StrOutputParser()

# Invoke the chain
result = chain.invoke(
    {"language": language or args.language, "task": task or args.task}
)

print(result)


######################################################################

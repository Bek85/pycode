import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import argparse
from langchain_core.runnables import RunnablePassthrough


parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="python")
parser.add_argument("--task", type=str, default="return the sum of two numbers")
parser.add_argument("--extension", type=str, default="py")
args = parser.parse_args()


# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

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
        code=lambda x: (code_prompt | llm).invoke(
            {"language": x["language"], "task": x["task"]}
        )
    )
    .assign(
        test=lambda x: (test_prompt | llm).invoke(
            {"language": x["language"], "code": x["code"]}
        )
    )
)

# Invoke the chain
response = chain.invoke({"language": args.language, "task": args.task})


print(">>>>>>>>>> GENERATED CODE <<<<<<<<<<")
print(response["code"])
print(">>>>>>>>>> GENERATED TEST <<<<<<<<<<")
print(response["test"])

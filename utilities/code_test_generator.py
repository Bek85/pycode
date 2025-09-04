import os
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
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI
from langchain.prompts import PromptTemplate
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="python")
parser.add_argument("--task", type=str, default="return the sum of two numbers")
parser.add_argument("--extension", type=str, default="py")
args = parser.parse_args()


# Model names now handled by config

######################################################################


# Load environment variables from .env file
load_dotenv()


######################################################################
# LangChain

llm = get_llm("local")

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

# Create chains for code and test generation
code_chain = code_prompt | llm
test_chain = test_prompt | llm

# Generate code first
input_data = {"language": language or args.language, "task": task or args.task}
generated_code = code_chain.invoke(input_data).content

# Generate test using the generated code
generated_test = test_chain.invoke({
    "language": input_data["language"], 
    "code": generated_code
}).content

# Prepare response
response = {
    "code": generated_code,
    "test": generated_test
}


print(">>>>>>>>>> GENERATED CODE <<<<<<<<<<")
print(response["code"])
print(">>>>>>>>>> GENERATED TEST <<<<<<<<<<")
print(response["test"])

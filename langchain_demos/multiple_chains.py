from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
# Handle both direct execution and module import
try:
    from ..config import get_llm
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_llm
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
import argparse
from operator import itemgetter

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="python")
parser.add_argument("--task", type=str, default="return the sum of two numbers")
args = parser.parse_args()

load_dotenv()

output_parser = StrOutputParser()

llm = get_llm("remote")

code_prompt = PromptTemplate(
    template="""
    You are a code generator.
    You are given a task to generate function for {language} that will {task}.
    """,
    input_variables=["language", "task"],
)

test_prompt = PromptTemplate(
    input_variables=["language", "content"],
    template="""
    You are a code tester.
    You are given code in {language}:
    {content}

    Generate comprehensive unit tests for this code.
    """,
)

# Get user input for language and task
language = input("Enter the programming language: ")
task = input("Enter the task description: ")

# Use command line args as fallback if no input provided
final_language = language or args.language
final_task = task or args.task

code_chain = code_prompt | llm | output_parser
test_chain = test_prompt | llm | output_parser


# Helper function to map code to content for test chain
def map_code_to_content(data):
    return {"language": data["language"], "content": data["code"]}


# Create a chain that combines code generation and test generation
chain = (
    # Step 1: Generate code (keeps original inputs + adds code)
    RunnablePassthrough.assign(code=code_chain)
    |
    # Step 2: Generate tests (keeps everything + adds test)
    RunnablePassthrough.assign(test=RunnableLambda(map_code_to_content) | test_chain)
)

# Invoke the chain with the inputs
result = chain.invoke({"language": final_language, "task": final_task})

# Format and print the output
print("\nGenerated Code:")
print("=" * 50)
print(result["code"])

print("\nGenerated Tests:")
print("=" * 50)
print(result["test"])
print("=" * 50)

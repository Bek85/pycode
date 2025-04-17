from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
import argparse
from operator import itemgetter

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="python")
parser.add_argument("--task", type=str, default="return the sum of two numbers")
args = parser.parse_args()


load_dotenv()

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

code_prompt = PromptTemplate(
    template="""
    You are a code generator.
    You are given a task to generate function for {language} that will {task}.
    """,
    input_variables=["language", "task"]
)

test_prompt = PromptTemplate(
    input_variables=["language", "content"],
    template="""
    You are a code tester.
    You are given code in {language}:
    {content}

    Generate comprehensive unit tests for this code.
    """
)

# Get user input for language and task
language = input("Enter the programming language: ")
task = input("Enter the task description: ")

# Use command line args as fallback if no input provided
final_language = language or args.language
final_task = task or args.task

# Create a chain that combines code generation and test generation
chain = (
    {
        "language": itemgetter("language"),
        "task": itemgetter("task")
    }
    | code_prompt
    | llm
    | (lambda code_response: {
        "language": final_language,
        "content": code_response.content
    })
    | test_prompt
    | llm
)

# Invoke the chain with the inputs
result = chain.invoke({
    "language": final_language,
    "task": final_task
})

# Format and print the output
print("\nGenerated Code and Tests:")
print("=" * 50)
content = result.content if hasattr(result, 'content') else str(result)
formatted_content = content.strip()
print(formatted_content)
print("=" * 50)

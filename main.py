import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="python")
parser.add_argument("--task", type=str, default="return the sum of two numbers")
parser.add_argument("--extension", type=str, default="py")
args = parser.parse_args()



# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

code_prompt = PromptTemplate(
  template="""
  You are a helpful assistant that can write code in {language}.
  Write a function that will {task}.
  """,
  input_variables=["language", "task"]
)

# Chain the prompt and the model
code_chain = code_prompt | llm

# Invoke the chain
response = code_chain.invoke({
  "language": args.language,
  "task": args.task
})

## Save the response to a file
# create a file extension based on the language
file_extension = "py" if args.language == "python" else args.extension

# create a file name based on the language and task
file_name = f"{args.language}_task.{file_extension}"


with open(file_name, "w") as f:
  f.write(response)

print(response)

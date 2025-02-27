import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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
  "language": "Python",
  "task": "return the sum of two numbers"
})

## Save the response to a file
with open("code.py", "w") as f:
  f.write(response)

print(response)

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

code_chain = LLMChain(
  llm=llm,
  prompt=code_prompt
)

response = code_chain.invoke({
  "language": "Python",
  "task": "return the sum of two numbers"
})

print(response)

import os
from dotenv import load_dotenv
from langchain_openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

response = llm.invoke('Write a short poem in Uzbek')

print(response)

from dotenv import load_dotenv
from ..config import get_llm  # Commented out as this file uses OpenAI SDK instead
import os
from openai import OpenAI  # OpenAI SDK

load_dotenv()

# Model names now handled by config if using LangChain
local_model_name = "ProkuraturaAI"  # local model name for OpenAI SDK
remote_model_name = "gpt-4o-mini"  # remote model name for OpenAI SDK


######################################################################
# Using OpenAI SDK
client = OpenAI(
    base_url="http://172.18.35.123:8000/v1",  # with base_url, you can override the default base url (https://api.openai.com/v1)
    api_key=os.getenv("OPENAI_API_KEY"),
)
# print(client.models.list())

user_input = input("Enter a prompt: ")

result = client.chat.completions.create(
    model=local_model_name,
    messages=[{"role": "user", "content": user_input}],
)
print(result.choices[0].message.content)

######################################################################


######################################################################
# Using LangChain

# llm = get_llm("local")

# user_input = input("Enter a prompt: ")

# result = llm.invoke(user_input)

# print(result.content)
######################################################################

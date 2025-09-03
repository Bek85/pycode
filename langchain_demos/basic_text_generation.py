from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os
from openai import OpenAI  # OpenAI SDK

load_dotenv()

local_model_name = "ProkuraturaAI"  # local model name
remote_model_name = "gpt-4o-mini"  # remote model name


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

# local_llm = init_chat_model(
#     model=local_model_name,
#     model_provider="openai",
#     openai_api_base="http://172.18.35.123:8000/v1",
# )

# remote_llm = init_chat_model(
#     model=remote_model_name,
#     model_provider="openai",
# )

# user_input = input("Enter a prompt: ")

# result = local_llm.invoke(user_input)

# print(result.content)
######################################################################

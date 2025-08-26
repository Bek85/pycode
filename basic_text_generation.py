from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

# from langchain_openai import (
#     OpenAI,
# )  # new import syntax for OpenAI. With this import, you can not specify the model name and other parameters

from openai import OpenAI

load_dotenv()

# client = OpenAI()
# print(client.models.list())


local_chat_model = init_chat_model(
    model="ProkuraturaAI",
    model_provider="openai",
    openai_api_base="http://172.18.35.123:8000/v1",
)

remote_chat_model = init_chat_model(
    model="gpt-5",  # chat-completions compatible
    model_provider="openai",
    # openai_api_base="https://api.openai.com/v1",  # <-- force OpenAI, not local
)


user_input = input("Enter a prompt: ")

result = local_chat_model.invoke(user_input)

print(result.content)

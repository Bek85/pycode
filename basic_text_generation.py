from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

# from langchain_openai import (
#     OpenAI,
# )  # new import syntax for OpenAI. With this import, you can not specify the model name and other parameters

from openai import OpenAI

load_dotenv()


local_chat_model = ChatOpenAI(
    model="llama3-8b-8192",
    openai_api_base="http://127.0.0.1:1234/v1",
)

remote_chat_model = init_chat_model(model="llama3-8b-8192", model_provider="openai")


user_input = input("Enter a prompt: ")

result = local_chat_model.invoke(user_input)

print(result.content)

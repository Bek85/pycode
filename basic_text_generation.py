from dotenv import load_dotenv
from langchain_openai import OpenAI # new import syntax for OpenAI. With this import, you can not specify the model name and other parameters.

load_dotenv()

llm = OpenAI()

result = llm.invoke(input("Ask anything: "))
print(result)

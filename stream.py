from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate

load_dotenv()

llm = init_chat_model("gpt-4o-mini", model_provider="openai", streaming=True)

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{content}"),
    ]
)

# chain = prompt | llm

# chain.invoke({"content": "Hello, how are you?"})

messages = prompt.format_messages(content="Tell me a joke")

output = llm.invoke(messages)

print(output)

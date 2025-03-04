from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
from langchain.memory import (
    ConversationBufferMemory,
    FileChatMessageHistory,
    ConversationSummaryMemory,
)

load_dotenv()

chat = ChatOpenAI(
    model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"), verbose=True
)

memory = ConversationSummaryMemory(
    llm=chat,
    chat_memory=FileChatMessageHistory(file_path="chat_history.json"),
    memory_key="chat_history",
    return_messages=True,
)

prompt = ChatPromptTemplate(
    input_variables=["content", "chat_history"],
    messages=[
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)


chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)


while True:
    user_input = input(">> ")
    result = chain.invoke({"content": user_input})

    print(result["text"])

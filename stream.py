from uuid import UUID
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import time

from langchain.chains import LLMChain

from langchain_core.outputs import ChatGenerationChunk, GenerationChunk

load_dotenv()


class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        pass


llm = init_chat_model(
    "gpt-4o-mini", model_provider="openai", streaming=True, callbacks=[StreamHandler()]
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{content}"),
    ]
)


class StreamingChain(LLMChain):
    def stream(self, input):
        print("hi there")


chainTest = StreamingChain(llm=llm, prompt=prompt)

chainTest.stream("asdfasdf")


# chain = prompt | llm

# Generate content from user input

# userInput = input("Ask anything: ")

# messages = prompt.format_messages(content=userInput)

# output = chain.stream(messages)

# print("Generating...")

# for chunk in output:
#     print(chunk.content, end="", flush=True)


# Print the output as it comes in with some delay and continue the conversation until the user exits the program with a keyboard interrupt

# while True:
#     for chunk in output:
#         print(chunk.content, end="", flush=True)
#         time.sleep(0.1)

#     userInput = input("\n\n----------------------------------\n\nAsk anything: ")
#     messages = prompt.format_messages(content=userInput)
#     output = chain.stream(messages)

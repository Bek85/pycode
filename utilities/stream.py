from uuid import UUID
from dotenv import load_dotenv
# Handle both direct execution and module import
try:
    from ..config import get_llm
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_llm
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import time

from langchain.chains import LLMChain

from langchain_core.outputs import ChatGenerationChunk, GenerationChunk

load_dotenv()


class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        pass


# Note: streaming and callbacks need to be configured separately when using get_llm
llm = get_llm("remote")
# llm = llm.with_callbacks([StreamHandler()])
# For streaming, you may need to configure the model appropriately

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

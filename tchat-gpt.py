from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate


prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[HumanMessagePromptTemplate.from_template("{content}")],
)


while True:
    user_input = input(">> ")
    print(f"You entered: {user_input}")

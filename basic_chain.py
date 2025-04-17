from dotenv import load_dotenv
from langchain_openai import OpenAI # new import syntax for OpenAI. With this import, you can not specify the model name and other parameters.
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model


load_dotenv()

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

code_prompt = PromptTemplate(
    template="""
    You are a code generator.
    You are given a task to generate function for {language} that will {task}.
    """,
    input_variables=["language", "task"]
)

# Get user input for language and task
language = input("Enter the programming language: ")
task = input("Enter the task description: ")

# Create a chain
chain = code_prompt | llm

# Invoke the chain
result = chain.invoke({"language": language, "task": task})

# Format the output in a readable way
print("\nGenerated Code:")
print("=" * 50)
# Extract and clean the content from the response
content = result.content if hasattr(result, 'content') else str(result)
# Remove any leading/trailing whitespace and format
formatted_content = content.strip()
print(formatted_content)
print("=" * 50)

from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# Initialize components
local_model_name = "ProkuraturaAI"
remote_model_name = "gpt-4o-mini"

# Initialize the LLM
llm = init_chat_model(
    model=local_model_name,
    model_provider="openai",
    openai_api_base="http://172.18.35.123:8000/v1",
)
output_parser = StrOutputParser()

# Create individual chains (reusable components)
code_prompt = PromptTemplate(
    input_variables=["task", "language"],
    template="Write a very short {language} function that will {task}.",
)

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}",
)

# Create atomic chains
code_chain = code_prompt | llm | output_parser
test_chain = test_prompt | llm | output_parser


# RECOMMENDED APPROACH: Enhanced RunnablePassthrough with error handling
def create_robust_sequential_chain():
    """
    Creates a sequential chain with proper error handling and logging.
    This approach offers the best balance of flexibility, maintainability, and scalability.
    """

    def generate_test_with_context(context):
        """Helper function to generate test with proper context"""
        try:
            return test_chain.invoke(
                {"language": context["language"], "code": context["code"]}
            )
        except Exception as e:
            print(f"Error generating test: {e}")
            return f"# Error generating test: {e}"

    # The main sequential chain using RunnablePassthrough
    sequential_chain = (
        # Step 1: Pass through initial inputs and generate code
        RunnablePassthrough.assign(
            code=lambda x: code_chain.invoke(
                {"task": x["task"], "language": x["language"]}
            )
        )
        # Step 2: Generate test using the code from step 1
        | RunnablePassthrough.assign(test=RunnableLambda(generate_test_with_context))
        # Step 3: Clean up output (optional transformation step)
        | RunnableLambda(
            lambda x: {
                "code": x["code"].strip(),
                "test": x["test"].strip(),
                "language": x["language"],
                "task": x["task"],
            }
        )
    )

    return sequential_chain


# ALTERNATIVE: More modular approach for complex workflows
class SequentialChainBuilder:
    """
    Builder pattern for creating complex sequential chains.
    Best for highly scalable applications with many steps.
    """

    def __init__(self, llm):
        self.llm = llm
        self.steps = []
        self.output_parser = StrOutputParser()

    def add_step(self, step_name, prompt_template, input_mapping=None):
        """Add a step to the sequential chain"""
        chain = prompt_template | self.llm | self.output_parser

        def step_function(context):
            if input_mapping:
                inputs = {k: context[v] for k, v in input_mapping.items()}
            else:
                inputs = context
            return chain.invoke(inputs)

        self.steps.append((step_name, step_function))
        return self

    def build(self):
        """Build the final sequential chain"""

        def execute_chain(initial_context):
            context = initial_context.copy()

            for step_name, step_function in self.steps:
                try:
                    result = step_function(context)
                    context[step_name] = result
                except Exception as e:
                    print(f"Error in step {step_name}: {e}")
                    context[step_name] = f"Error: {e}"

            return context

        return RunnableLambda(execute_chain)


# Usage of the builder pattern
def create_builder_chain():
    builder = SequentialChainBuilder(llm)

    return (
        builder.add_step("code", code_prompt, {"task": "task", "language": "language"})
        .add_step("test", test_prompt, {"language": "language", "code": "code"})
        .build()
    )


# SIMPLE LCEL for straightforward cases
simple_sequential = RunnablePassthrough.assign(
    code=code_chain
) | RunnablePassthrough.assign(
    test=lambda x: test_chain.invoke({"language": x["language"], "code": x["code"]})
)

# Choose your approach:
if __name__ == "__main__":
    # RECOMMENDED: Use this for most production cases
    sequential_chain = create_robust_sequential_chain()

    # OR for very complex workflows:
    # sequential_chain = create_builder_chain()

    # OR for simple cases:
    # sequential_chain = simple_sequential

    # Execute
    result = sequential_chain.invoke({"language": args.language, "task": args.task})

    print(">>>>>> GENERATED CODE:")
    print(result["code"])

    print(">>>>>> GENERATED TEST:")
    print(result["test"])

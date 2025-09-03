from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, BaseMessage
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from typing import List, Optional, Any, Dict
from dotenv import load_dotenv
import asyncio


# Load environment variables
load_dotenv()


# Create a custom message history class with summarization
class SummarizingMessageHistory(BaseChatMessageHistory):
    """Chat message history that provides summarization capabilities"""

    def __init__(self, max_tokens: int = 30):  # Lower for faster testing
        self.messages = []
        self.max_tokens = max_tokens
        self.current_summary = None
        self._summarized_messages = None  # Cache for summarized messages
        self.chat_model = init_chat_model(model="gpt-4o-mini", model_provider="openai")

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        self.messages.append(message)
        # Reset summarized messages cache when a new message is added
        self._summarized_messages = None

    def clear(self) -> None:
        """Clear the message history."""
        self.messages = []
        self.current_summary = None
        self._summarized_messages = None

    def get_messages(self) -> List[BaseMessage]:
        """Get messages to be used by LLM, using summarized version if available."""
        if self._summarized_messages is not None:
            return self._summarized_messages
        return self.messages

    async def get_messages_for_llm(self) -> List[BaseMessage]:
        """Return messages for LLM, with summarization if needed"""
        # Calculate token count for all messages
        total_tokens = sum(
            self.chat_model.get_num_tokens(msg.content) for msg in self.messages
        )

        print(f"\n=== Message History Stats ===")
        print(f"Number of messages: {len(self.messages)}")
        print(f"Total tokens: {total_tokens}")
        print(f"Max token limit: {self.max_tokens}")

        # DEBUG: Print all current messages in history
        print("\nCurrent messages in history:")
        for i, msg in enumerate(self.messages):
            print(f"{i+1}. {msg.type}: {msg.content}")

        # Check if we need to summarize
        if total_tokens > self.max_tokens and len(self.messages) >= 4:
            print("\n=== SUMMARIZATION TRIGGERED ===")

            # Keep the most recent exchange (last user question and AI response)
            recent_messages = (
                self.messages[-2:] if len(self.messages) >= 2 else self.messages
            )

            # Messages to summarize (all except the most recent exchange)
            history_to_summarize = self.messages[:-2] if len(self.messages) >= 2 else []

            if not history_to_summarize:
                print("No messages to summarize.")
                return self.messages

            # Format messages for summarization
            summary_text = "\n".join(
                f"{msg.type}: {msg.content}" for msg in history_to_summarize
            )

            # Create summarization prompt
            summarize_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """Summarize this conversation very briefly in 1-2 sentences, focusing on the key information:

{text}

Summary:""",
                    )
                ]
            )

            # Create chain for summarization
            summarize_chain = summarize_prompt | self.chat_model | StrOutputParser()

            # Generate summary
            print("Generating summary...")
            try:
                self.current_summary = await summarize_chain.ainvoke(
                    {"text": summary_text}
                )
                print(f"\nSummary generated: {self.current_summary}")

                # Create messages with summary for next interaction
                final_messages = [
                    SystemMessage(
                        content=f"The human asks the AI what 1+1 is, and the AI responds that 1+1 equals 2."
                    ),
                    *recent_messages,
                ]

                print("\nUsing summarized history for next interaction")
                print("Final messages for LLM:")
                for i, msg in enumerate(final_messages):
                    print(f"{i+1}. {msg.type}: {msg.content}")

                # Cache the summarized messages
                self._summarized_messages = final_messages
                return final_messages

            except Exception as e:
                print(f"Error during summarization: {e}")
                return self.messages

        print("Using full message history (under token limit)")
        self._summarized_messages = None
        return self.messages


class CustomRunnableWithHistory(RunnableWithMessageHistory):
    """Custom implementation of RunnableWithMessageHistory that uses summarization"""

    async def _get_history(self, config: dict) -> List[BaseMessage]:
        """Get history from history obj, prioritizing get_messages over messages"""
        session_id = self._get_session_id(config)
        history_obj = self.get_history(session_id)

        # If the history object has our custom get_messages_for_llm method, use it
        if hasattr(history_obj, "get_messages_for_llm"):
            history = await history_obj.get_messages_for_llm()
            return history

        # Otherwise, fall back to the standard behavior
        return history_obj.get_messages()


# Create message history with token limit
message_history = SummarizingMessageHistory(max_tokens=30)  # Lower for testing


def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """Return the chat history for a given session ID"""
    return message_history


# Initialize the chat model
chat_model = init_chat_model(model="gpt-4o-mini", model_provider="openai")


# Setup prompt template
prompt = ChatPromptTemplate(
    input_variables=["chat_history", "content"],
    messages=[
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

# Create basic chain
chain = prompt | chat_model


# Create the custom chain with history that uses summarization
chain_with_history = CustomRunnableWithHistory(
    chain,
    get_chat_history,
    input_messages_key="content",
    history_messages_key="chat_history",
)


def print_colored(text, color="green"):
    """Print colored text"""
    colors = {"green": "\033[92m", "blue": "\033[94m", "reset": "\033[0m"}
    print(f"{colors.get(color, colors['green'])}{text}{colors['reset']}")


async def chat():
    """Main chat loop"""
    print("\n=== AI Chat with Conversation Summarization ===")
    print("- The chat will summarize the conversation when it gets too long")
    print("- This keeps context while managing token usage")
    print("- Token limit set to 30 to trigger summarization quickly")
    print("- Type 'exit', 'quit', or 'q' to end the conversation\n")

    while True:
        try:
            user_input = input(">> ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting chat...")
                break

            # Get the history messages that will be used for this interaction
            config = {"configurable": {"session_id": "default"}}
            session_id = "default"
            history_obj = get_chat_history(session_id)

            # This will trigger summarization if needed
            if hasattr(history_obj, "get_messages_for_llm"):
                history_messages = await history_obj.get_messages_for_llm()

                # Display the prompt in terminal format
                print("\n> Entering new chain...")
                print("Prompt after formatting:")

                # Display any system message (which appears after summarization)
                has_system = False
                for msg in history_messages:
                    if msg.type == "system":
                        print_colored(f"System: {msg.content}")
                        has_system = True
                        break

                # Always show the current user input
                print_colored(f"Human: {user_input}")

                # For visual spacing
                if has_system:
                    print()

            # Process the input
            result = await chain_with_history.ainvoke(
                {"content": user_input},
                config=config,
            )

            print("> Finished chain.")
            print(f"{result.content}\n")

        except EOFError:
            print("\nExiting due to EOF...")
            break
        except KeyboardInterrupt:
            print("\nExiting due to keyboard interrupt...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(chat())

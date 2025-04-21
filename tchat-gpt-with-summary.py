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
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import asyncio


load_dotenv()


# Initialize the chat model first
chat_model = init_chat_model(model="gpt-4o-mini", model_provider="openai")


class DebugCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self._last_chain_input = None

    def on_llm_start(self, serialized, messages, **kwargs):
        print("\n========= Final LLM Input =========")
        token_count = sum(chat_model.get_num_tokens(str(msg)) for msg in messages)
        print(f"Token count: {token_count}")
        print(f"Number of messages: {len(messages)}")
        print("\nActual content being sent to LLM:")
        for msg in messages:
            print(f"\n{msg}")
        print("=================================\n")

    def on_chain_start(self, serialized, inputs, **kwargs):
        # Avoid duplicate logs by checking the input content
        current_input = (
            f"{inputs.get('content', '')}{str(inputs.get('chat_history', ''))}"
        )
        if current_input == self._last_chain_input:
            return
        self._last_chain_input = current_input

        print("\n=== Chain Input ===")
        if "chat_history" in inputs:
            history_messages = inputs["chat_history"]
            if history_messages:
                print(f"\nChat History ({len(history_messages)} messages):")
                total_tokens = sum(
                    chat_model.get_num_tokens(msg.content) for msg in history_messages
                )
                print(f"Total history tokens: {total_tokens}")
                for msg in history_messages:
                    print(f"{msg.type}: {msg.content}")
            else:
                print("No chat history yet")
        if "content" in inputs:
            print(f"\nCurrent Input: {inputs['content']}")
        print("==================\n")


# Create debug flag
DEBUG_MODE = True

# Initialize callback handler for debugging
debug_callbacks = [DebugCallbackHandler()] if DEBUG_MODE else []

# Update the chat model with callbacks
chat_model = init_chat_model(
    model="gpt-4o-mini", model_provider="openai", callbacks=debug_callbacks
)


# Create a custom message history class
class SummarizingMessageHistory(BaseChatMessageHistory):
    """Chat message history that provides summarization capabilities"""

    def __init__(self, max_tokens: int = 30):
        self.messages = []
        self.max_tokens = max_tokens
        self.current_summary = None

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        self.messages.append(message)

    def clear(self) -> None:
        """Clear the message history."""
        self.messages = []
        self.current_summary = None

    async def get_messages_for_llm(self) -> List[BaseMessage]:
        """Return messages for LLM, with summarization if needed"""
        # Calculate token count for all messages
        total_tokens = sum(
            chat_model.get_num_tokens(msg.content) for msg in self.messages
        )

        print(f"\n=== Message History Stats ===")
        print(f"Number of messages: {len(self.messages)}")
        print(f"Total tokens: {total_tokens}")
        print(f"Max token limit: {self.max_tokens}")

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
            summarize_chain = summarize_prompt | chat_model | StrOutputParser()

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
                        content=f"Previous conversation summary: {self.current_summary}"
                    ),
                    *recent_messages,
                ]

                print("\nUsing summarized history for next interaction")
                return final_messages

            except Exception as e:
                print(f"Error during summarization: {e}")
                return self.messages

        print("Using full message history (under token limit)")
        return self.messages


# Create message history with token limit
message_history = SummarizingMessageHistory(max_tokens=30)


def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """Return the chat history for a given session ID"""
    return message_history


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


class SummarizingChainWithHistory(RunnableWithMessageHistory):
    """A custom chain with history that handles summarization"""

    async def _get_history(self, config: dict) -> List[BaseMessage]:
        """Override to use get_messages_for_llm if available"""
        session_id = self._get_session_id(config)
        history = self.get_history(session_id)

        # Use custom get_messages_for_llm method if available
        if hasattr(history, "get_messages_for_llm"):
            messages = await history.get_messages_for_llm()
            return messages

        # Fall back to standard behavior
        return history.messages


# Create chain with history
chain_with_history = SummarizingChainWithHistory(
    chain,
    get_chat_history,
    input_messages_key="content",
    history_messages_key="chat_history",
)


async def chat():
    """Main chat loop"""
    print("\n=== Math Chat with Summarization ===")
    print("Token limit set to 30 - summarization will trigger after a few messages")
    print("Try asking math questions like 'what is 1+1?' and build on them\n")

    while True:
        try:
            user_input = input(">> ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting chat...")
                break

            result = await chain_with_history.ainvoke(
                {"content": user_input},
                config={
                    "configurable": {"session_id": "default"},
                    "callbacks": debug_callbacks,
                },
            )
            print("\nAssistant:", result.content)

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

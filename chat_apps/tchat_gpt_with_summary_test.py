from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
# Handle both direct execution and module import
try:
    from ..config import get_llm
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_llm
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from typing import List, Optional, Any, Dict, Callable
from dotenv import load_dotenv
import asyncio
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="tchat-gpt-with-summary-test.log",
    filemode="w",
)
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()


# Initialize the chat model first
llm = get_llm("remote")


class DebugCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self._last_chain_input = None

    def on_llm_start(self, serialized, messages, **kwargs):
        log_message = "\n========= Final LLM Input =========\n"
        token_count = sum(llm.get_num_tokens(str(msg)) for msg in messages)
        log_message += f"Token count: {token_count}\n"
        log_message += f"Number of messages: {len(messages)}\n"
        log_message += "\nActual content being sent to LLM:\n"
        for msg in messages:
            log_message += f"\n{msg}\n"
        log_message += "=================================\n"
        logger.info(log_message)
        print(log_message)

    def on_chain_start(self, serialized, inputs, **kwargs):
        # Avoid duplicate logs by checking the input content
        current_input = (
            f"{inputs.get('content', '')}{str(inputs.get('chat_history', ''))}"
        )
        if current_input == self._last_chain_input:
            return
        self._last_chain_input = current_input

        log_message = "\n=== Chain Input ===\n"
        if "chat_history" in inputs:
            history_messages = inputs["chat_history"]
            if history_messages:
                log_message += f"\nChat History ({len(history_messages)} messages):\n"
                total_tokens = sum(
                    llm.get_num_tokens(msg.content) for msg in history_messages
                )
                log_message += f"Total history tokens: {total_tokens}\n"
                for msg in history_messages:
                    log_message += f"{msg.type}: {msg.content}\n"
            else:
                log_message += "No chat history yet\n"
        if "content" in inputs:
            log_message += f"\nCurrent Input: {inputs['content']}\n"
        log_message += "==================\n"
        logger.info(log_message)
        print(log_message)


# Create debug flag
DEBUG_MODE = True

# Initialize callback handler for debugging
debug_callbacks = [DebugCallbackHandler()] if DEBUG_MODE else []

# Update the chat model with callbacks
llm = get_llm("remote")
# Note: callbacks need to be added separately if needed
# llm = llm.with_callbacks(debug_callbacks)


# Create a custom message history class
class SummarizingMessageHistory(BaseChatMessageHistory):
    """Chat message history that provides summarization capabilities"""

    def __init__(self, max_tokens: int = 30):
        self.messages = []
        self.max_tokens = max_tokens
        self.current_summary = None
        self._summarized_messages = None  # Cache for summarized messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        self.messages.append(message)
        # Reset summarized messages cache when a new message is added
        self._summarized_messages = None
        logger.info(f"Added message: {message.type} - {message.content}")

    def clear(self) -> None:
        """Clear the message history."""
        self.messages = []
        self.current_summary = None
        self._summarized_messages = None
        logger.info("Message history cleared")

    def get_messages(self) -> List[BaseMessage]:
        """Get messages to be used by LLM, using summarized version if available."""
        if self._summarized_messages is not None:
            logger.info("Using cached summarized messages")
            return self._summarized_messages
        return self.messages

    async def get_messages_for_llm(self) -> List[BaseMessage]:
        """Return messages for LLM, with summarization if needed"""
        # Calculate token count for all messages
        total_tokens = sum(llm.get_num_tokens(msg.content) for msg in self.messages)

        log_message = f"\n=== Message History Stats ===\n"
        log_message += f"Number of messages: {len(self.messages)}\n"
        log_message += f"Total tokens: {total_tokens}\n"
        log_message += f"Max token limit: {self.max_tokens}\n"
        logger.info(log_message)
        print(log_message)

        # Check if we need to summarize
        if total_tokens > self.max_tokens and len(self.messages) >= 4:
            log_message = "\n=== SUMMARIZATION TRIGGERED ===\n"
            logger.info(log_message)
            print(log_message)

            # Keep the most recent exchange (last user question and AI response)
            recent_messages = (
                self.messages[-2:] if len(self.messages) >= 2 else self.messages
            )

            # Messages to summarize (all except the most recent exchange)
            history_to_summarize = self.messages[:-2] if len(self.messages) >= 2 else []

            if not history_to_summarize:
                logger.info("No messages to summarize.")
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
            summarize_chain = summarize_prompt | llm | StrOutputParser()

            # Generate summary
            logger.info("Generating summary...")
            print("Generating summary...")
            try:
                self.current_summary = await summarize_chain.ainvoke(
                    {"text": summary_text}
                )
                logger.info(f"Summary generated: {self.current_summary}")
                print(f"\nSummary generated: {self.current_summary}")

                # Create messages with summary for next interaction
                final_messages = [
                    SystemMessage(
                        content=f"Previous conversation summary: {self.current_summary}"
                    ),
                    *recent_messages,
                ]

                logger.info("Using summarized history for next interaction")
                logger.info(f"Final message count: {len(final_messages)}")
                for msg in final_messages:
                    logger.info(f"Final message: {msg.type} - {msg.content}")

                print("Using summarized history for next interaction")

                # Cache the summarized messages
                self._summarized_messages = final_messages
                return final_messages

            except Exception as e:
                logger.error(f"Error during summarization: {e}")
                print(f"Error during summarization: {e}")
                return self.messages

        logger.info("Using full message history (under token limit)")
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
            logger.info("Using custom get_messages_for_llm method")
            history = await history_obj.get_messages_for_llm()
            logger.info(f"Got {len(history)} messages from get_messages_for_llm")
            return history

        # Otherwise, fall back to the standard behavior
        logger.info("Using standard messages from history object")
        return history_obj.get_messages()


# Create message history with token limit
message_history = SummarizingMessageHistory(max_tokens=30)


def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """Return the chat history for a given session ID"""
    logger.info(f"Getting chat history for session: {session_id}")
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
chain = prompt | llm


# Create the custom chain with history that uses summarization
custom_chain_with_history = CustomRunnableWithHistory(
    chain,
    get_chat_history,
    input_messages_key="content",
    history_messages_key="chat_history",
)


async def run_test_conversation():
    """Run a test conversation with predetermined inputs"""
    logger.info("Starting test conversation")
    print("\n=== Math Chat with Summarization - Automated Test ===")
    print("Token limit set to 30 - summarization will trigger after a few messages")

    # Test conversation flow
    test_inputs = ["what is 1+1?", "and 3 more?", "and 10 more?"]

    for user_input in test_inputs:
        try:
            print(f"\n>> {user_input}")
            logger.info(f"User input: {user_input}")

            # Process the input with our custom chain
            result = await custom_chain_with_history.ainvoke(
                {"content": user_input},
                config={
                    "configurable": {"session_id": "default"},
                    "callbacks": debug_callbacks,
                },
            )

            print(f"\nAssistant: {result.content}")
            logger.info(f"Assistant response: {result.content}")

        except Exception as e:
            error_msg = f"Error during conversation: {str(e)}"
            logger.error(error_msg)
            print(f"\nERROR: {error_msg}")

    logger.info("Test conversation completed")
    print("\n=== Test completed ===")


if __name__ == "__main__":
    asyncio.run(run_test_conversation())

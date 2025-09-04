from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from typing import List
from dotenv import load_dotenv
from colorama import Fore, Style, init
import asyncio

# Handle both direct execution and module import
try:
    from ..config import get_llm
except ImportError:
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_llm

load_dotenv()

# Initialize colorama for Windows compatibility
init(autoreset=True)


class SummarizingMessageHistory(BaseChatMessageHistory):
    """Chat message history that summarizes old messages when token limit is exceeded."""

    def __init__(self, max_tokens: int = 30):
        self.messages = []
        self.max_tokens = max_tokens
        self.chat_model = get_llm("local")

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        self.messages.append(message)

    def clear(self) -> None:
        """Clear the message history."""
        self.messages = []

    def get_messages(self) -> List[BaseMessage]:
        """Return messages, implementing the required interface method."""
        return self.messages

    async def get_messages_for_llm(self) -> List[BaseMessage]:
        """Return messages for LLM, with summarization if needed."""
        total_tokens = sum(
            self.chat_model.get_num_tokens(msg.content) for msg in self.messages
        )

        print(
            f"Messages: {len(self.messages)}, Tokens: {total_tokens}/{self.max_tokens}"
        )

        # If under token limit or too few messages, return as-is
        if total_tokens <= self.max_tokens or len(self.messages) < 4:
            return self.messages

        return await self._create_summarized_history()

    async def _create_summarized_history(self) -> List[BaseMessage]:
        """Create a summarized version of the message history."""
        print("Summarizing conversation history...")

        # Keep the last 2 messages (recent user-assistant exchange)
        recent_messages = self.messages[-2:]
        messages_to_summarize = self.messages[:-2]

        if not messages_to_summarize:
            return self.messages

        # Create summary of old messages
        summary_text = "\n".join(
            f"{msg.type}: {msg.content}" for msg in messages_to_summarize
        )

        summary = await self._generate_summary(summary_text)

        # Return: [summary as system message] + [recent messages]
        return [
            SystemMessage(content=f"Previous conversation summary: {summary}"),
            *recent_messages,
        ]

    async def _generate_summary(self, text: str) -> str:
        """Generate a summary of the given text."""
        print("🤖 Generating summary with LLM...")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Summarize this conversation briefly in 1-2 sentences, "
                    "focusing on key information:\n\n{text}\n\nSummary:",
                )
            ]
        )

        chain = prompt | self.chat_model | StrOutputParser()

        try:
            summary = await chain.ainvoke({"text": text})
            print("✅ Summary generated successfully")
            return summary
        except Exception as e:
            print(f"❌ Summarization failed: {e}")
            return (
                "Previous conversation context unavailable due to summarization error."
            )


class CustomRunnableWithHistory(RunnableWithMessageHistory):
    """Custom wrapper that uses our summarizing message history."""

    async def _get_history(self, config: dict) -> List[BaseMessage]:
        """Get history, using summarization if available."""
        print("🔍 _get_history() called - checking for summarization...")
        session_id = self._get_session_id(config)
        history_obj = self.get_history(session_id)

        if hasattr(history_obj, "get_messages_for_llm"):
            return await history_obj.get_messages_for_llm()

        return history_obj.get_messages()


# Global message history instance
# Set enable_summarization=False if using problematic local models
message_history = SummarizingMessageHistory(max_tokens=30)


def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """Return the chat history for a given session ID."""
    return message_history


def setup_chat_chain():
    """Set up the chat chain with history and summarization."""
    llm = get_llm("local")

    prompt = ChatPromptTemplate(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{content}"),
        ]
    )

    chain = prompt | llm

    return CustomRunnableWithHistory(
        chain,
        get_chat_history,
        input_messages_key="content",
        history_messages_key="chat_history",
    )


async def chat():
    """Main chat loop."""
    print(f"{Fore.CYAN}🤖 AI Chat with Conversation Summarization{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Token limit set to 30 for quick summarization demo{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Type 'exit', 'quit', or 'q' to end{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Type 'debug' to see current message history{Style.RESET_ALL}\n")

    chain_with_history = setup_chat_chain()
    config = {"configurable": {"session_id": "default"}}

    while True:
        try:
            user_input = input(f"{Fore.GREEN}👤 You: {Style.RESET_ALL}")

            if user_input.lower() in ["exit", "quit", "q"]:
                print(f"{Fore.CYAN}👋 Goodbye!{Style.RESET_ALL}")
                break

            if user_input.lower() == "debug":
                print("\n=== DEBUG INFO ===")
                history = get_chat_history("default")
                print(f"Current messages in history: {len(history.messages)}")
                for i, msg in enumerate(history.messages):
                    token_count = history.chat_model.get_num_tokens(msg.content)
                    print(
                        f"  {i+1}. {msg.type}: {msg.content[:50]}... ({token_count} tokens)"
                    )
                total_tokens = sum(
                    history.chat_model.get_num_tokens(msg.content)
                    for msg in history.messages
                )
                print(f"Total tokens: {total_tokens}")
                print("=== END DEBUG ===\n")
                continue

            # IMPORTANT: Check history before processing
            history_obj = get_chat_history("default")
            if hasattr(history_obj, "get_messages_for_llm"):
                print("🔍 Checking if summarization is needed...")
                await history_obj.get_messages_for_llm()

            # Process input and get response
            result = await chain_with_history.ainvoke(
                {"content": user_input}, config=config
            )

            llm = get_llm("local")
            print(f"{Fore.BLUE}🤖 {getattr(llm, 'model_name', None) or getattr(llm, 'model', 'AI')}: {Style.RESET_ALL}{result.content}")
            print(f"{Fore.YELLOW}{'-'*50}{Style.RESET_ALL}\n")

        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.CYAN}👋 Goodbye!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(chat())

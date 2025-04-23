from langchain.callbacks.base import BaseCallbackHandler
from pyboxen import boxen


def boxen_print(*args, **kwargs):
    print(boxen(*args, **kwargs))


class ChatModelStartHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("\n\n=============== Sending Messages to LLM ===============\n\n")

        for message in messages[0]:
            # Get the message type name without the "Message" suffix if possible
            msg_type = message.__class__.__name__.replace("Message", "").lower()

            # Handle both AIMessage and AIMessageChunk with the same logic
            if msg_type.startswith("ai"):
                if (
                    hasattr(message, "additional_kwargs")
                    and "function_call" in message.additional_kwargs
                ):
                    call = message.additional_kwargs["function_call"]
                    boxen_print(
                        f"Running tool {call['name']} with arguments {call['arguments']}",
                        title="AI Function Call",
                        color="cyan",
                    )
                else:
                    # Handle both empty and non-empty AI messages
                    content = (
                        message.content if message.content else "[Awaiting response...]"
                    )
                    boxen_print(content, title="AI", color="blue")

            elif msg_type == "system":
                boxen_print(message.content, title="System", color="yellow")

            elif msg_type == "human":
                boxen_print(message.content, title="Human", color="green")

            elif msg_type == "function":
                boxen_print(message.content, title="Function Result", color="purple")

            else:
                # Fallback for any other message types
                boxen_print(
                    message.content, title=f"{msg_type.capitalize()}", color="white"
                )

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
)

user_input = input("Enter a prompt: ")

while True:
    user_input = input("Enter a prompt: ")
    response = client.chat.completions.create(
        model="qwen3:4b",
        messages=[{"role": "user", "content": user_input}],
    )

    print(response.choices[0].message.content)

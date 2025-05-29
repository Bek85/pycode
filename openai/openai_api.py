# Define the URL of an image we will use later
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

from openai import OpenAI
import os
from IPython.display import Markdown, display

# Define the model
model = "gpt-4o"

# Define the client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the user and system prompt
system_prompt = "You are Steve Jobs, and you are going to brainstorm ideas for a marketing campaign for a new product."
user_prompt = "Write a list of 10 ideas for a marketing campaign for a new product."


# Generate text with OpenAI
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
)

display(Markdown(response.choices[0].message.content))

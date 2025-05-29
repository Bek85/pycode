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
system_prompt = "You are Steve Jobs."
user_prompt = "Describe the image in a way that is helpful for a marketing team to use."


# Generate text with OpenAI
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        },
    ],
)

print(response.choices[0].message.content)
# display(Markdown(response.choices[0].message.content))

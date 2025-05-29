# Define the URL of an image we will use later
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

from openai import OpenAI
import os
from IPython.display import Markdown, display
import base64

file_name = "Thumbnail python FV1.jpg"
file_path = os.path.join(os.getcwd(), "openai", file_name)

# Read the image and convert to base64
with open(file_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{image_base64}"

# Define the model
model = "gpt-4o"

# Define the client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the user and system prompt
system_prompt = "You are a helpful assistant."
user_content = [
    {
        "type": "text",
        "text": "Describe the image.",
    },
    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
]


# Generate text with OpenAI
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_content,
        },
    ],
)

print(response.choices[0].message.content)
# display(Markdown(response.choices[0].message.content))

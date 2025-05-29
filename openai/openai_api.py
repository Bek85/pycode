# Define the URL of an image we will use later
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

from openai import OpenAI
import os

# Define the model
model = "gpt-4oi"

# Define the client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

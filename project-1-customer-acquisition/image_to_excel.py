import os
from openai import OpenAI

model = "gpt-4o"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

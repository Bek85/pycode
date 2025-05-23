# We are using GPT2 because

# *   it is free
# *   We can use the transformers library

# In later sections we will use the openAI API (paid)


# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "Dear, boss ... "


# Simplified text generation function
def generate_text(prompt, model, tokenizer, max_length=100):
    # Encode the prompt to get input IDs
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text using the model
    output = model.generate(input_ids, max_length=100)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


# Generate text
generated_text = generate_text(prompt, model, tokenizer)
print(generated_text)

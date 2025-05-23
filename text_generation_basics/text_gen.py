# We are using GPT2 because

# *   it is free
# *   We can use the transformers library

# In later sections we will use the openAI API (paid)


# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sample_data import data


# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "Dear, boss ... "

# Tokenization
# All inputs must have the same length
# Add a dummy token to the end
# Having the same length => this is called padding
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the data
tokenized_data = [
    tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        return_tensors="pt",
        padding="max_length",
        max_length=50,
    )
    for sentence in data
]

# print(tokenized_data[:2])

# Isolate the input IDs and attention masks
input_ids = [item["input_ids"].squeeze() for item in tokenized_data]
attention_masks = [item["attention_mask"].squeeze() for item in tokenized_data]

# Convert the input IDs and attention masks to tensors
# This step is necessary for processing the tuned model
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)

# Padding all sequences to make sure they are the same length
padded_input_ids = pad_sequence(
    input_ids, batch_first=True, padding_value=tokenizer.eos_token_id
)

padded_attention_masks = pad_sequence(
    attention_masks, batch_first=True, padding_value=0
)


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
# print(generated_text)

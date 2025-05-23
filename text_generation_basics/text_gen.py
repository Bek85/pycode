# We are using GPT2 because

# *   it is free
# *   We can use the transformers library

# In later sections we will use the openAI API (paid)


# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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
input_ids = torch.stack(input_ids)
attention_masks = torch.stack(attention_masks)

# Padding all sequences to make sure they are the same length
padded_input_ids = pad_sequence(
    input_ids, batch_first=True, padding_value=tokenizer.eos_token_id
)

padded_attention_masks = pad_sequence(
    attention_masks, batch_first=True, padding_value=0
)


# Improved text generation function with proper attention mask
def generate_text(prompt, model, tokenizer, max_new_tokens=100):
    # Encode the prompt properly with attention mask
    encoded = tokenizer.encode_plus(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Generate text using the model with attention mask
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


# Generate text
generated_text = generate_text(prompt, model, tokenizer)
# print(generated_text)

from torch.utils.data import DataLoader
from custom_dataset_class import dataset
import torch

# Import the pre-configured tokenizer and model from text_gen.py
from text_gen import tokenizer, model

# Prepare data in batches
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Set the model to training mode
model.train()

# Training loop
for epoch in range(10):
    for batch in dataloader:
        # Get the input IDs and attention masks IDs
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_masks"]
        labels = batch["labels"]

        # Reset the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        # Backward pass: compute the gradients of the loss
        loss = outputs.loss
        loss.backward()

        # Update the weights
        optimizer.step()

        # Print the loss
        # print(f"Epoch {epoch}, Loss: {loss.item()}")


def generate_text_with_finetuned_model(prompt, model, tokenizer, max_length=100):
    # Encode the prompt to get the input_ids
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate text using the model
    output = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_length=max_length
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# Test the function

prompt = "In this research, we"

generated_text = generate_text_with_finetuned_model(
    prompt, model, tokenizer, max_length=500
)

print(generated_text)

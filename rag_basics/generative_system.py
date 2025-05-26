from transformers import AutoModelForCausalLM, AutoTokenizer
from retreival_system_with_FAISS import retrieved_documents

# Initialize the generative tokenizer and model
generative_tokenizer = AutoTokenizer.from_pretrained("gpt2")
generative_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Set the pad token to the eos token
generative_tokenizer.pad_token = generative_tokenizer.eos_token

# Define the context and question
context = " ".join(retrieved_documents)
question = "What is Berlinale?"
# Prompt
prompt = f"Context: {context} \n Question: {question} \n Answer:"


# Create a function to generate text
def generate_text(context, query, model, tokenizer, max_length=100):
    # Format the input text with context and query
    input_text = f"Context {context}\nQuestion: {query}\nAnswer:"

    # Tokenize the input text and prepare tensors for the model
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    input_ids = inputs["input_ids"]
    attention_masks = (input_ids != tokenizer.pad_token_id).long()

    # Generate text using the model
    outputs = model.generate(
        input_ids,
        attention_mask=attention_masks,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.1,  # Controls randomness
        top_k=50,  # Controls diversity
        top_p=0.8,  # Controls diversity, different from top_k is that top_p is a probability threshold
        repetition_penalty=1.2,  # Penalizes repetition
        do_sample=True,  # Sample from the model
    )

    # Decode the generated text to a readable format
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

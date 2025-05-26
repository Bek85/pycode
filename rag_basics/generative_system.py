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

    # Tokenize the prompt
    inputs = generative_tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True
    )

    input_ids = inputs["input_ids"]

    attention_masks = (input_ids != generative_tokenizer.pad_token_id).long()

    outputs = generative_model.generate(
        input_ids,
        attention_mask=attention_masks,
        max_length=100,
        pad_token_id=generative_tokenizer.eos_token_id,
    )

    answer = generative_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


generated_answer = generate_text(
    context, question, generative_model, generative_tokenizer
)

# print(generated_answer)

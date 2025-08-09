# chat.py
# A simple command-line interface to chat with trained SMT model.
# VERSION 2: Includes temperature and top-k sampling for better generation.

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import os

# Import the model architecture from model.py file
from model import GuidanceInfusedTransformer

# --- Configuration ---
# Ensure these match the parameters used during training in pretrain.py
MODEL_PARAMS = {
    "vocab_size": 32128,
    "d_model": 512,
    "n_layers": 6,
    "n_heads": 8,
    "d_ff": 2048,
    "guidance_dim": 128
}

# IMPORTANT: Update this path to exact checkpoint file.
CHECKPOINT_PATH = "./checkpoints/model_epoch_1_batch_10000.pt"

# --- Generation Parameters ---
MAX_NEW_TOKENS = 20 # The maximum number of tokens to generate
TEMPERATURE = 0.7    # Controls randomness. Higher is more random. (0.7-0.9 is a good range)
TOP_K = 50           # Considers only the top K most likely tokens for sampling.

# --- UPDATED GENERATION FUNCTION ---
def generate_response(model, tokenizer, prompt, device):
    """
    Generates a response from the model given a prompt using temperature and top-k sampling.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = input_ids.transpose(0, 1)

    generated_ids = input_ids
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            # Get logits from the model
            outputs = model(generated_ids)
            next_token_logits = outputs[-1, :, :] # Logits for the last token

            # 1. Apply Temperature
            # This softens the probability distribution, making less likely tokens more probable
            scaled_logits = next_token_logits / TEMPERATURE
            
            # 2. Apply Top-k Sampling
            # This filters the logits to only the top K most likely tokens
            top_k_logits, top_k_indices = torch.topk(scaled_logits, TOP_K)
            
            # Convert filtered logits to probabilities
            probabilities = F.softmax(top_k_logits, dim=-1)
            
            # Sample from the filtered distribution
            next_token_index_in_top_k = torch.multinomial(probabilities, num_samples=1)
            next_token_id = torch.gather(top_k_indices, -1, next_token_index_in_top_k)

            # Append the sampled token to the sequence
            generated_ids = torch.cat([generated_ids, next_token_id.transpose(0,1)], dim=0)

            # Stop if the end-of-sequence token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    response_text = tokenizer.decode(generated_ids.transpose(0, 1).squeeze(), skip_special_tokens=True)
    return response_text

def main():
    """
    Main function to load the model and start the chat loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at '{CHECKPOINT_PATH}'")
        print("Please make sure the path is correct and that trained the model.")
        return

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    MODEL_PARAMS['vocab_size'] = tokenizer.vocab_size

    print("Loading model architecture...")
    model = GuidanceInfusedTransformer(**MODEL_PARAMS).to(device)
    
    print(f"Loading weights from checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")

    print("\n--- GIT Chat Interface (v2) ---")
    print("Using temperature and top-k sampling for less repetitive output.")
    print("Type 'exit' to quit.")
    print("-" * 25)

    while True:
        prompt = input("You: ")
        if prompt.lower() == 'exit':
            break
        
        response = generate_response(model, tokenizer, prompt, device)
        print(f"GIT Model: {response}")
        print("-" * 25)

if __name__ == "__main__":
    main()

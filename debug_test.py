# debug_test.py
# A diagnostic script to verify the model's ability to learn.
# It attempts to overfit on a single batch of data.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import time

# Import the model architecture and dataset class from our other files
from model import GuidanceInfusedTransformer
from pretrain import PretrainDataset

# --- Configuration ---
# Using the same parameters as main training script
MODEL_PARAMS = {
    "vocab_size": 32128,
    "d_model": 512,
    "n_layers": 6,
    "n_heads": 8,
    "d_ff": 2048,
    "guidance_dim": 128
}
TRAINING_PARAMS = {
    "learning_rate": 5e-4,
    "batch_size": 8,
    "seq_length": 256,
    "num_workers": 2
}

def run_overfit_test():
    """
    Trains the model on a single batch repeatedly to check if the loss decreases.
    """
    print("--- Starting Overfit Diagnostic Test ---")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    MODEL_PARAMS['vocab_size'] = tokenizer.vocab_size

    # Initialize a fresh, untrained Model and Optimizer
    model = GuidanceInfusedTransformer(**MODEL_PARAMS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMS["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Prepare Dataset and get just ONE batch
    print("Loading one batch from Wikipedia dataset...")
    wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split='train', streaming=True)
    dataset_for_one_batch = PretrainDataset(wiki_dataset, tokenizer, TRAINING_PARAMS["seq_length"])
    
    # We create a dataloader just to easily get one formatted batch
    dataloader = DataLoader(
        dataset_for_one_batch,
        batch_size=TRAINING_PARAMS["batch_size"],
        num_workers=TRAINING_PARAMS["num_workers"]
    )
    
    # Get the single batch we will overfit on
    try:
        single_batch = next(iter(dataloader))
    except Exception as e:
        print(f"Error getting a batch. This might be a dataloader issue on Windows.")
        print("Try setting num_workers=0 in TRAINING_PARAMS and run again.")
        print(f"Original error: {e}")
        return

    inputs, targets = single_batch
    inputs = inputs.transpose(0, 1).to(device)
    targets = targets.transpose(0, 1).to(device)
    print(f"Successfully loaded one batch of data with shape: {inputs.shape}")

    model.train()
    
    print("\n--- Starting Overfitting Loop (Training on the same batch 300 times) ---")
    print("The loss should decrease dramatically and approach zero.")
    
    for i in range(1, 301):
        start_time = time.time()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1, MODEL_PARAMS['vocab_size']), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        if i % 20 == 0: # Print loss every 20 steps
            print(f"Step {i:3d} | Loss: {loss.item():.4f} | Time: {(time.time() - start_time)*1000:.2f}ms")

    print("\n--- Test Finished ---")
    if loss.item() < 1.0:
        print("✅ SUCCESS: The loss dropped significantly. The model architecture is capable of learning.")
        print("The issue is likely that pre-training just requires more time and data.")
    else:
        print("❌ FAILURE: The loss did not decrease significantly.")
        print("This indicates a potential fundamental issue in the model architecture or training loop.")

if __name__ == "__main__":
    run_overfit_test()

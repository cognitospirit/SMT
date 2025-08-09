# pretrain.py
# Stage 1: Foundational pre-training for the Self-Module Transformer (SMT)
# This script includes a Gradio GUI for controlling the training process.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import gradio as gr
import time
import os
import json

# Import the model architecture from model.py file
from model import GuidanceInfusedTransformer

# --- Configuration ---
# can adjust these hyperparameters
MODEL_PARAMS = {
    "vocab_size": 32128,  # Vocabulary size for a standard tokenizer like SentencePiece
    "d_model": 512,
    "n_layers": 6,
    "n_heads": 8,
    "d_ff": 2048,
    "guidance_dim": 128
}
TRAINING_PARAMS = {
    "learning_rate": 5e-4,
    "batch_size": 8,       # Adjust based on VRAM
    "seq_length": 256,     # Length of sequences to train on
    "num_workers": 2,      # For the dataloader
    "checkpoint_dir": "./checkpoints",
    "state_file": "./checkpoints/training_state.json"
}

# Global state to control training from the GUI
training_should_continue = False

# --- Dataset Handling ---
class PretrainDataset(IterableDataset):
    """
    An iterable dataset that yields tokenized sequences from a streaming dataset.
    """
    def __init__(self, dataset, tokenizer, seq_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            if 'text' in example and example['text']:
                tokens = self.tokenizer.encode(example['text'])
                buffer.extend(tokens)
                while len(buffer) >= self.seq_length + 1:
                    seq = buffer[:self.seq_length + 1]
                    # The input is the sequence, the target is the sequence shifted by one
                    input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                    target_ids = torch.tensor(seq[1:], dtype=torch.long)
                    yield input_ids, target_ids
                    buffer = buffer[self.seq_length:]

# --- Checkpointing Functions ---
def save_checkpoint(model, optimizer, processed_batches, epoch):
    """Saves model, optimizer, and training state."""
    os.makedirs(TRAINING_PARAMS["checkpoint_dir"], exist_ok=True)
    
    # Save model and optimizer state
    checkpoint_path = os.path.join(TRAINING_PARAMS["checkpoint_dir"], f"model_epoch_{epoch}_batch_{processed_batches}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    # Save training progress
    with open(TRAINING_PARAMS["state_file"], 'w') as f:
        json.dump({"processed_batches": processed_batches, "last_checkpoint": checkpoint_path}, f)
    
    return f"Checkpoint saved to {checkpoint_path}"

def load_checkpoint(model, optimizer):
    """Loads the most recent checkpoint if it exists."""
    if not os.path.exists(TRAINING_PARAMS["state_file"]):
        return 0, "No checkpoint found. Starting from scratch."

    with open(TRAINING_PARAMS["state_file"], 'r') as f:
        state = json.load(f)
    
    checkpoint_path = state.get("last_checkpoint")
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        processed_batches = state.get("processed_batches", 0)
        return processed_batches, f"Resumed from checkpoint: {checkpoint_path}"
    else:
        return 0, "State file found, but checkpoint file is missing. Starting over."

# --- The Main Training Function ---
def train_model(progress=gr.Progress()):
    """The main function that runs the training loop."""
    global training_should_continue
    training_should_continue = True
    
    yield "--- Initializing Training ---"
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yield f"Using device: {device}"

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    MODEL_PARAMS['vocab_size'] = tokenizer.vocab_size

    # Initialize Model and Optimizer
    model = GuidanceInfusedTransformer(**MODEL_PARAMS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMS["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint if available
    processed_batches, status_message = load_checkpoint(model, optimizer)
    yield status_message

    # Prepare Dataset
    yield "Loading Wikipedia dataset (streaming)..."
    wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split='train', streaming=True)
    
    # *** CORRECTED RESUME LOGIC ***
    # Skip batches that have already been processed
    if processed_batches > 0:
        yield f"Skipping {processed_batches} already processed articles..."
        # Use the .skip() method which is pickle-safe for multiprocessing
        resumed_dataset = wiki_dataset.skip(processed_batches * TRAINING_PARAMS['batch_size'])
        dataset_to_train = PretrainDataset(resumed_dataset, tokenizer, TRAINING_PARAMS["seq_length"])
    else:
        dataset_to_train = PretrainDataset(wiki_dataset, tokenizer, TRAINING_PARAMS["seq_length"])

    dataloader = DataLoader(
        dataset_to_train,
        batch_size=TRAINING_PARAMS["batch_size"],
        num_workers=TRAINING_PARAMS["num_workers"]
    )

    model.train()
    running_loss = 0.0
    log_interval = 100
    checkpoint_interval = 1000
    
    yield "--- Starting Training Loop ---"
    
    current_batch_num = processed_batches
    for inputs, targets in dataloader:
        current_batch_num += 1
        if not training_should_continue:
            yield "Training paused by user."
            break

        start_time = time.time()
        
        inputs = inputs.transpose(0, 1).to(device)
        targets = targets.transpose(0, 1).to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs.reshape(-1, MODEL_PARAMS['vocab_size']), targets.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        running_loss += loss.item()
        end_time = time.time()
        
        if current_batch_num % log_interval == 0:
            avg_loss = running_loss / log_interval
            tokens_per_sec = (TRAINING_PARAMS['batch_size'] * TRAINING_PARAMS['seq_length']) / (end_time - start_time)
            status_update = f"Batch {current_batch_num} | Loss: {avg_loss:.4f} | Tokens/sec: {tokens_per_sec:.0f}"
            yield status_update
            running_loss = 0.0

        if current_batch_num % checkpoint_interval == 0:
            status_update = save_checkpoint(model, optimizer, current_batch_num, epoch=1)
            yield status_update
            
    if training_should_continue:
        yield "--- Training Finished ---"
    else:
        status_update = save_checkpoint(model, optimizer, current_batch_num, epoch=1)
        yield f"Final state saved before pausing: {status_update}"


def stop_training():
    global training_should_continue
    training_should_continue = False
    return "Stop signal sent. Training will pause after the current batch."

# --- Gradio GUI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Guidance-Infused Transformer (GIT) - Pre-training")
    gr.Markdown("Stage 1: Foundational training on the Wikipedia dataset. Press 'Start / Resume' to begin.")

    with gr.Row():
        start_button = gr.Button("Start / Resume Training", variant="primary")
        stop_button = gr.Button("Pause Training")

    status_textbox = gr.Textbox(label="Training Status", lines=20, interactive=False)

    start_button.click(
        fn=train_model,
        inputs=[],
        outputs=status_textbox
    )
    stop_button.click(
        fn=stop_training,
        inputs=[],
        outputs=status_textbox
    )

if __name__ == "__main__":
    os.makedirs(TRAINING_PARAMS["checkpoint_dir"], exist_ok=True)
    demo.launch()

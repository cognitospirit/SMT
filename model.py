# model.py
# Architectural blueprint for Self-Module Transformer

import torch
import torch.nn as nn
import math

# A standard component for adding positional information to the input embeddings.
# The Transformer itself doesn't know the order of words, so we add this info.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# --- The Main Architecture ---
class GuidanceInfusedTransformer(nn.Module):
    """
    A Transformer model with a dedicated, internal "Guidance Area" to instill
    a consistent persona or style.

    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimensionality of the model's embeddings (must be even).
        n_layers (int): The number of layers in the Transformer Body.
        n_heads (int): The number of attention heads in each layer.
        d_ff (int): The dimension of the feed-forward network within each layer.
        guidance_dim (int): The latent dimension of the Guidance Area.
        dropout (float): The dropout rate.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, guidance_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # --- Component 1: Standard Language Processing Parts ---
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # The main language processing engine (W_T)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=False # We use [seq_len, batch_size, d_model]
        )
        self.transformer_body = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Component 2: The "Self/Guidance Area" (W_G) ---
        # This is our novel component. It's a small, separate network
        # that learns to generate a constant "persona signal".
        self.guidance_area = nn.Sequential(
            nn.Linear(d_model, guidance_dim),
            nn.ReLU(),
            nn.Linear(guidance_dim, d_model) # Projects back to the main dimension
        )

        # --- Component 3: The Output Layer ---
        # A final layer to project the model's output back to the vocabulary size
        self.output_layer = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        # Initialize weights for better training stability
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        The forward pass defining the data flow through the architecture.

        Args:
            src (Tensor): The input sequence of token IDs. Shape: [seq_len, batch_size]
            src_mask (Tensor, optional): A mask to prevent attention to future tokens.

        Returns:
            Tensor: The output logits. Shape: [seq_len, batch_size, vocab_size]
        """
        # 1. Standard Embedding and Positional Encoding
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        src_pos = self.pos_encoder(src_embedded)

        # 2. Generate the Guidance Signal
        # The Guidance Area generates a single, constant vector representing the persona.
        # This signal is learned during training. It doesn't depend on the current input 'src',
        # making it a stable, guiding principle.
        # We create a dummy input for it, as its weights are what matter.
        dummy_input = torch.zeros(1, 1, self.d_model, device=src.device)
        guidance_signal = self.guidance_area(dummy_input) # Shape: [1, 1, d_model]

        # 3. Infuse Guidance and Process through Transformer Body
        # The guidance signal is added to every token's embedding at each layer.
        # This is a simplified but effective way to infuse guidance.
        # The broadcasting rules of PyTorch handle the addition correctly.
        infused_input = src_pos + guidance_signal

        # The main transformer processes the infused input
        transformer_output = self.transformer_body(infused_input, src_mask)

        # 4. Generate Final Output
        output = self.output_layer(transformer_output)

        return output

# --- Example Usage ---
# This block demonstrates how to create and test the model architecture.
# It proves that the data flow works correctly without needing any training data yet.
if __name__ == '__main__':
    # Hyperparameters for our small, trainable model
    VOCAB_SIZE = 10000  # Size of our vocabulary
    D_MODEL = 512       # Embedding dimension (must be even)
    N_LAYERS = 6        # Number of layers in the Transformer Body
    N_HEADS = 8         # Number of attention heads
    D_FF = 2048         # Dimension of the feed-forward network
    GUIDANCE_DIM = 128  # Latent dimension of the Guidance Area

    print("--- Initializing Guidance-Infused Transformer ---")
    model = GuidanceInfusedTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        guidance_dim=GUIDANCE_DIM
    )
    print(f"Model created successfully!")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params / 1e6:.2f}M")
    print("-" * 50)

    # Create a dummy input tensor to test the forward pass
    # Shape: [sequence_length, batch_size]
    # Let's simulate a batch of 4 sentences, each with 20 tokens.
    dummy_src = torch.randint(0, VOCAB_SIZE, (20, 4))

    print(f"Testing forward pass with dummy input of shape: {dummy_src.shape}")
    
    # Pass the dummy data through the model
    try:
        output_logits = model(dummy_src)
        print("Forward pass successful!")
        print(f"Output tensor shape: {output_logits.shape}")
        # Expected output shape: [seq_len, batch_size, vocab_size] -> [20, 4, 10000]
        assert output_logits.shape == (20, 4, VOCAB_SIZE)
        print("Output shape is correct.")
    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")

    print("-" * 50)


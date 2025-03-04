import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        token_emb = self.token_embedding(x)  # (B, T, d_model)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))  # (T, d_model)
        x = token_emb + pos_emb.unsqueeze(0)
        x = self.transformer(x, x)  # Self-attention
        logits = self.lm_head(x)
        return logits

# Define model hyperparameters
vocab_size = 50257  # For example, like GPT-2
d_model = 256
num_heads = 8
num_layers = 6
block_size = 128

# Initialize model
model = GPT(vocab_size, d_model, num_heads, num_layers, block_size)


# --- Training ---

# Fix: Ensure data is in (batch_size, sequence_length) shape
seq_length = 32  # Choose a reasonable sequence length

def train(model, data, epochs=10, batch_size=32, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for i in range(0, len(data) - batch_size * seq_length, batch_size * seq_length):
            x = data[i:i + batch_size * seq_length].view(batch_size, seq_length)
            y = data[i + 1:i + 1 + batch_size * seq_length].view(batch_size, seq_length)

            optimizer.zero_grad()
            print("Input shape:", x.shape)  # Should print (batch_size, seq_length)

            logits = model(x)

            # Reshape logits and targets for loss calculation
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Dummy dataset (random tokens)
data = torch.randint(0, vocab_size, (10000,))

# Train model
train(model, data)
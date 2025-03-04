import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import datetime
import tiktoken


# Load text data
with open("text_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding("cl100k_base")
vocab_size = len(tokenizer.encoder)

# Encode & Decode functions
def encode(text):
    return tiktoken.encode(text)

def decode(indices):
    return tiktoken.decode(text)

# Convert text to tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split into train and validation
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

# Define model parameters
batch_size = 64
block_size = 128  # Context size for GPT
embedding_dim = 256
num_heads = 4
num_layers = 6
dropout = 0.0

# Define GPT-like Model
class CharGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, block_size, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        self.block_size = block_size

    def forward(self, x):
        b, t = x.shape
        token_emb = self.token_embedding(x)  # (B, T, E)
        pos_emb = self.position_embedding(torch.arange(t, device=x.device))  # (T, E)
        x = token_emb + pos_emb  # (B, T, E)
        x = self.transformer(x)  # (B, T, E)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

# Create dataset
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y

# Function to save the model with timestamp
def save_model(model, optimizer, epoch, loss):
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Define the filename with timestamp
    filename = f"model_{timestamp}_epoch_{epoch+1}_loss_{loss.item():.4f}.pth"
    
    # Save model state dict and optimizer state dict
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    
    print(f"Model saved as {filename}")

# Text Generation
def generate_text(start_text, max_length=250):
    model.eval()
    context = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0).to(device)
    generated = context.tolist()[0]

    for _ in range(max_length):
        logits = model(context)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_char = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_char)
        context = torch.tensor([generated[-block_size:]], dtype=torch.long).to(device)
    
    return decode(generated)

# Training loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CharGPT(vocab_size, embedding_dim, num_heads, num_layers, block_size, dropout).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.1)

epochs = 1000
for epoch in range(epochs):
    model.train()
    x, y = get_batch('train')
    x, y = x.to(device), y.to(device)
    
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save model at the end of each epoch
    if (epoch + 1) % 50 == 0:  # Save every n epochs
        # save_model(model, optimizer, epoch, loss)

        # Generate sample text

        generated_text = generate_text("\"")

        # Print to console and append to file
        with open('output.txt', 'a', encoding='utf-8') as f:
            print(generated_text)  # Print to console
            f.write(f"Epoch {epoch+1}: Loss = {loss.item():.4f}"+ '\n')
            f.write(generated_text + '\n')  # Append to file

    
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")


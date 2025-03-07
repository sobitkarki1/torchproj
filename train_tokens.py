import tiktoken
import numpy as np
from tqdm import tqdm

# Load your training text
with open("text_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Convert text into bytes (BPE operates on byte sequences)
text_bytes = text.encode("utf-8")

# Define vocab size (smaller for local testing)
vocab_size = 20000  # Adjust as needed

# Train BPE tokenizer using tiktoken's training function
new_encoder = tiktoken.bpe.train_bytes(
    text_bytes,  # Training data
    vocab_size=vocab_size,  # Desired vocabulary size
    n_threads=4  # Adjust based on CPU cores
)

# Save the trained model
with open("custom_bpe.tiktoken", "wb") as f:
    f.write(new_encoder.to_bytes())

print(f"Tokenizer trained with vocab size {vocab_size}")

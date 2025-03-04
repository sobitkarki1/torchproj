import tiktoken
import json

# Load tokenizer
enc = tiktoken.get_encoding("cl100k_base")

# Create vocabulary mapping
vocab = {}

for i in range(enc.n_vocab):
    try:
        decoded = enc.decode([i])  # Try decoding token
        vocab[i] = decoded
    except KeyError:
        vocab[i] = f"<UNDECODABLE_TOKEN_{i}>"  # Handle undecodable tokens safely

# Save to JSON file
with open("vocabulary.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)

print("Saved vocabulary.json successfully!")

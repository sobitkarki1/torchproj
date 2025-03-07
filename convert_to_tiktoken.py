import json
import tiktoken

# Load the saved BPE tokenizer from tokenizers
with open("bpe_tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

# Extract merges (byte pair rules) and vocabulary
bpe_merges = [(pair[0], pair[1]) for pair in tokenizer_data["model"]["merges"]]
vocab = tokenizer_data["model"]["vocab"]

# Convert vocab into a list of tuples (token, index)
vocab_items = sorted(vocab.items(), key=lambda item: item[1])
vocab_list = [item[0] for item in vocab_items]

# Create a tiktoken-compatible BPE model
enc = tiktoken.Encoding(
    name="custom-bpe",
    pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    mergeable_ranks={word: i for i, word in enumerate(vocab_list)},
    special_tokens={"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
)

# Save it to a file
enc.save("tiktoken_bpe.json")

# Load the tokenizer
enc = tiktoken.get_encoding("custom-bpe")

# Test encoding
test_text = "Hello, how are you?"
tokens = enc.encode(test_text)
print("Token IDs:", tokens)
print("Decoded:", enc.decode(tokens))

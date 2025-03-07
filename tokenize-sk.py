from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

# Initialize BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# Define pre-tokenization (splitting text into words)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Load dataset
with open("text_data.txt", "r", encoding="utf-8") as f:
    data = f.read().splitlines()

# Train tokenizer on dataset
trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
tokenizer.train_from_iterator(data, trainer)

# Post-processing (adding special tokens)
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))],
)

# Save tokenizer
tokenizer.save("bpe_tokenizer.json")

# Load tokenizer
tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

# Test tokenization
test_text = "Hello, how are you?"
encoded = tokenizer.encode(test_text)
print("Tokens:", encoded.tokens)
print("IDs:", encoded.ids)

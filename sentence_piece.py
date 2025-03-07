import sentencepiece as spm

# Train a tokenizer on your multilingual dataset
spm.SentencePieceTrainer.Train(
    input="text_data.txt", 
    model_prefix="tokenizer", 
    vocab_size=10000, 
    character_coverage=0.9995, 
    model_type="bpe"
)

# Load the tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

# Tokenize text
text = "This is a test. 这是一个测试。Esto es una prueba."
tokens = sp.encode_as_pieces(text)
print(tokens)

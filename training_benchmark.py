from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from datasets import load_dataset
import torch

model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = load_dataset("imdb", split="train[:1%]").shuffle(seed=42).select(range(500))

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

trainer.train()

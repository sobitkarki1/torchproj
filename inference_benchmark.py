from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_id = "gpt2"  # Start with small model; try "EleutherAI/gpt-neo-1.3B" later
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {model_id} on {device}")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=False)
model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=False).to(device)
print("Model Load Time:", round(time.time() - t0, 2), "s")

# Run inference
prompt = "Once upon a time,"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
t0 = time.time()
outputs = model.generate(**inputs, max_new_tokens=20)
print("Inference Time:", round(time.time() - t0, 2), "s")
print("Output:", tokenizer.decode(outputs[0]))

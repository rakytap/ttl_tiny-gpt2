import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from ttl_gpt2 import GPT2LMHeadModelTTL

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
model.eval()

print(type(model))

text = "Once upon a time"
inputs = tokenizer(text, return_tensors="pt")


with torch.no_grad():
    outputs = model(**inputs)

# print(outputs)

logits = outputs.logits
next_token = torch.argmax(logits[0, -1])
print(tokenizer.decode(next_token))


# Groq implementation
model_ttl = GPT2LMHeadModelTTL(model.config)

# Convert PyTorch tensors to numpy arrays
inputs_numpy = {key: value.numpy() for key, value in inputs.items()}
model_ttl.compile_ttl(**inputs_numpy)

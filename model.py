import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from transformers import GPT2Tokenizer


if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

try:
     model = torch.load('/home/jabulani/NLP_Project/model.pt', map_location=torch.device('cpu'))
     tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
except Exception as e:
    print(f'Loading model error: {e.args}')


def generate(prompt, len_gen=200, temperature=0.8):
    try:
        generated = tokenizer.encode(prompt)
        context = torch.tensor([generated]).to(device)
        past = None

        for i in range(len_gen):
            output, past = model(context, past_key_values=past).values()
            output = output / temperature
            token = torch.distributions.Categorical(logits=output[..., -1, :]).sample()
            generated += token.tolist()
            context = token.unsqueeze(0)
        sequence = tokenizer.decode(generated)
    except Exception as e:
        print(f"Error: {e.args}")
    print(sequence)
    return sequence

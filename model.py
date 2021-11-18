import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from transformers import GPT2Tokenizer

    
def generate(prompt, len_gen=200, temperature=0.8):
    if torch.cuda.is_available():    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    try:
        model = torch.load('model.pt', map_location=torch.device('cpu'))
        tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
        generated = tokenizer.encode(prompt)
        context = torch.tensor([generated]).to(device)
        past = None

        for i in range(len_gen):
            print(f'Iter-{i} in {time.strftime("%M:%S", time.localtime())}: Get values for out and past')
            output, past = model(context, past_key_values=past).values()
            output = output / temperature
            print(f'Iter-{i} in {time.strftime("%M:%S", time.localtime())}: Token distributions')
            token = torch.distributions.Categorical(logits=output[..., -1, :]).sample()
            print(f'Iter-{i} in {time.strftime("%M:%S", time.localtime())}: Token to list')
            generated += token.tolist()
            print(f'Iter-{i} in {time.strftime("%M:%S", time.localtime())}: Token unsqueeze')
            context = token.unsqueeze(0)
        print(f'Iter-{len_gen} in {time.asctime()}: Decode tokenizer')
        sequence = tokenizer.decode(generated)
    except Exception as e:
        print(f"Error: {e.args}")
    return sequence



# def exp(x):
#     if not isinstance(x, float):
#         try: 
#             x = float(x)
#             result = math.expm1(x)
#         except:
#             result = 'Not a number'
    
#     return result


# def draw_plot(inp):
#     x = np.random.normal(size=100, loc=-15)
#     # print(x)
#     plt.scatter(x, x**2 + np.random.normal(size=len(x)))
#     plt.savefig('plot.jpg')
#     return 'plot.jpg'

# def generate():
#     ...
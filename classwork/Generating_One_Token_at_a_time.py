"""
This python file is an exercise as a part of Generative AI lecture 
focusing on understanding how an LLM generates text, one token at a time,
by using the previous tokens to predict the following ones
"""

#Load Tokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer

#load Tokenizer using the HuggingFace
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

#Create A partial sentence

text = "Niggas be dying for the chain around"
inputs =  tokenizer(text, return_tensors= "pt")

#show tokens as numbers (input_ids)

inputs["inputs_ids"]


#Step 2.... Examine Tokenaization
import pandas as pd

def show_tokenization(inputs):
    return pd.DataFrame ([(id, tokenizer.decode(id)) for id in 
                          inputs["inputs_ids"][0]], columns = ["id", "token"],)

show_tokenization(inputs)


#step 2.2 calculate the probabilities for the next token for all possible choices. 

import torch 

with torch.no_grad():
    logits = model (**inputs).logits[:, -1, :]
    probabilities = torch.nn.functional.softmax(logits[0],dim =-1 )

def show_next_token_choices (probabilities, top_n=5):
    return pd.DataFrame(
    [
        (id, tokenizer.decode(id), p.item() for id, p in enumerate(probabilities) if p.item())
    ],
    columns=["id", "token","p"]


    ).sort_values("p", ascending=False)[:top_n]

show_next_token_choices(probabilities)



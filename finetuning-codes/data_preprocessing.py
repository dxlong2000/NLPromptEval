# from datasets import load_dataset
# import random 
# import csv 

# dataset = load_dataset("vicgalle/alpaca-gpt4")["train"]
# data = [dt for dt in dataset]
# random.shuffle(data)

# train_data = [[dt["instruction"], dt["output"]] for dt in data[:2500]]
# valid_data = [[dt["instruction"], dt["output"]] for dt in data[2500:2750]]

# with open("/home/long/WhatMakesAGoodPrompt/finetuning-codes/data/train_2500.csv", "w") as file:
#     csvwriter = csv.writer(file)
#     csvwriter.writerow(["prompt", "response"])
#     csvwriter.writerows(train_data)

# with open("/home/long/WhatMakesAGoodPrompt/finetuning-codes/data/valid_250.csv", "w") as file:
#     csvwriter = csv.writer(file)
#     csvwriter.writerow(["prompt", "response"])
#     csvwriter.writerows(valid_data)

import openai
import re
from tqdm import tqdm
import csv
import time
from tqdm import tqdm

WING_KEY = "YOUR-KEY-HERE"

def get_gpt4o_mini_answer(user_prompt, max_tokens=512, temperature=0.1, system_prompt="You are a helpful assistant!", n=1):
    api_token = WING_KEY
    model_name = "gpt-4o-mini"
    openai.api_key = api_token
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=n
    )
    if n==1: return response.choices[0].message.content
    else: return [response.choices[idx].message.content for idx in range(n)]

INPUT_PATH = "/home/WhatMakesAGoodPrompt/finetuning-codes/data/valid_250.csv"
OUTPUT_PATH = "/home/WhatMakesAGoodPrompt/finetuning-codes/data/formal_valid_250.csv"

saved_data = [] 
with open(INPUT_PATH) as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        prompt = row[0] 
        modified_prompt = "Please " + prompt[0].lower() + prompt[1:]
        
        saved_data.append([modified_prompt, row[1]])

with open(OUTPUT_PATH, "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(header)
    csvwriter.writerows(saved_data)
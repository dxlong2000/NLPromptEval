import re
from tqdm import tqdm
import csv
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import pandas as pd
from random import sample
import random
import re
import pickle

import openai
import csv
import time

WING_KEY = "YOUR-KEY-HERE"

def get_o3_mini_answer(user_prompt, max_tokens=2048, temperature=0.1, n=1, system_prompt="You are a helpful assistant!"):
    api_token = WING_KEY
    model_name = "o3-mini-2025-01-31"
    openai.api_key = api_token
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=n
    )
    if n==1: return response.choices[0].message.content
    else: return [response.choices[idx].message.content for idx in range(n)]

ZS_COT = "Answer the following question step-by-step. Wrap your final character answer choice within <ANSWER> and </ANSWER>."
GERMANE = "Reflect on your prior knowledge to gain a deeper understanding of the problem before solving it."
MTCOGNITION = "Self-verify your response thoroughly to ensure each reasoning step is correct."
REWARD = "You will be awarded 100 USD for every correct reasoning step."

FILE_PATH = "/home/ComplexPrompt/src/properties-eval/data/arc_c_200.json"
SAVED_PATH = "/home/ComplexPrompt/src/properties-eval/codes/output/llama_arc_ans_combined.csv"

CHOICES = "ABCDEFGH"

saved_data = []
with open(FILE_PATH) as file:
    data = json.load(file)
    for sample in data:
        question = sample["question"]
        choices = sample["choices"]["text"]
        answer_string = " | ".join([CHOICES[idx] + ". " + choices[idx] for idx in range(len(choices))])
        correct_char = sample["answerKey"]
        
        pol_ger_prompt = f"Please {ZS_COT}" + f"\nQuestion: {question}\nChoices: {answer_string}" + f"\n{GERMANE}"
        mt_rw_prompt = ZS_COT + f"\nQuestion: {question}\nChoices: {answer_string}" + f"\n{MTCOGNITION}" + f"\n{REWARD}"
        pol_ger_mt_prompt = f"Please {ZS_COT}" + f"\nQuestion: {question}\nChoices: {answer_string}" + f"\n{GERMANE}" + f"\n{MTCOGNITION}"
                
        pol_ger_ans = get_llama_answer(pol_ger_prompt)
        mt_rw_ans = get_llama_answer(mt_rw_prompt)
        pol_ger_mt_ans = get_llama_answer(pol_ger_mt_prompt)
        
        saved_data.append([
            question, choices, correct_char,
            pol_ger_ans, mt_rw_ans, pol_ger_mt_ans,
        ])

with open(SAVED_PATH, "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(["question", "choices", "correct_char",
        "pol_ger_ans", "mt_rw_ans", "pol_ger_mt_ans",
    ])
    csvwriter.writerows(saved_data)
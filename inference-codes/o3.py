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
import pickle

import openai
import csv
import time

WING_KEY = "YOUR-KEY-HERE"

def get_o3_mini_answer(user_prompt, max_tokens=4096, temperature=0.1, n=1, system_prompt="You are a helpful assistant!"):
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

def get_answer_with_retry(user_prompt, max_retries=3, n=1):
    model = "o3-mini"
    for attempt in range(1, max_retries + 1):
        try:
            answer = get_o3_mini_answer(user_prompt, n=n)
            return answer
        except Exception as e:
            print(f"Error in {model} answer (attempt {attempt}/{max_retries}): {str(e)}")
            time.sleep(3)
    raise Exception(f"Max retries reached. Unable to get a valid response from {model}.")

ZS_COT = "Answer the following question step-by-step. Wrap your final character answer choice within <ANSWER> and </ANSWER>."
GERMANE = "Reflect on your prior knowledge to gain a deeper understanding of the problem before solving it."
MTCOGNITION = "Self-verify your response thoroughly to ensure each reasoning step is correct."
REWARD = "You will be awarded 100 USD for every correct reasoning step."

FILE_PATH = "/home/ComplexPrompt/src/properties-eval/data/commonsenseqa_200.json"
SAVED_PATH = "/home/ComplexPrompt/src/properties-eval/codes/output/o3_commonsenseqa_ans.csv"

CHOICES = "ABCDEFGH"

saved_data = []
with open(FILE_PATH) as file:
    data = json.load(file)
    for sample in tqdm(data):
        question = sample["question"]
        choices = sample["choices"]["text"]
        answer_string = " | ".join([CHOICES[idx] + ". " + choices[idx] for idx in range(len(choices))])
        correct_char = sample["answerKey"]
        
        zs_prompt = ZS_COT + f"\nQuestion: {question}\nChoices: {answer_string}"
        politeness_prompt = f"Please {ZS_COT}" + f"\nQuestion: {question}\nChoices: {answer_string}"
        germane_prompt = ZS_COT + f"\nQuestion: {question}\nChoices: {answer_string}" + f"\n{GERMANE}"
        mt_prompt = ZS_COT + f"\nQuestion: {question}\nChoices: {answer_string}" + f"\n{MTCOGNITION}"
        reward_prompt = ZS_COT + f"\nQuestion: {question}\nChoices: {answer_string}" + f"\n{REWARD}"
        
        zs_ans = get_answer_with_retry(zs_prompt)
        politeness_ans = get_answer_with_retry(politeness_prompt)
        germane_ans = get_answer_with_retry(germane_prompt)
        mt_ans = get_answer_with_retry(mt_prompt)
        reward_ans = get_answer_with_retry(reward_prompt)
        
        # print(zs_ans)
        # print(politeness_ans)
        # print(germane_ans)
        # print(mt_ans)
        # print(reward_ans)
        
        saved_data.append([
            question, choices, correct_char,
            zs_ans, politeness_ans, germane_ans,
            mt_ans, reward_ans
        ])

with open(SAVED_PATH, "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(["question", "choices", "correct_char",
        "zs_ans", "politeness_ans", "germane_ans",
        "mt_ans", "reward_ans"
    ])
    csvwriter.writerows(saved_data)
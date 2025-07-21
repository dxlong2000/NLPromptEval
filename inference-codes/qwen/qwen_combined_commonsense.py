import pandas as pd
from random import sample
import random
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
from tqdm import tqdm
import pandas as pd
import csv 
import random
import re 

device = "cuda" # the device to load the model onto
model_id = "/disk2/Long/WhatMakesAGoodPrompt/finetuning-codes/output/qwen-2.5-7b-alpaca-instruct-2452025-ver12"
hf_token = "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"

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def get_qwen_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generate_ids = model.generate(inputs.input_ids, max_new_tokens=1024, do_sample=True, top_p = 0.9)
    answer = tokenizer.decode(generate_ids[0][len(inputs.input_ids[0]):]).strip()
    return answer.replace("<|endoftext|>", "")

ZS_COT = "Answer the following question step-by-step. Wrap your final answer choice within <ANSWER> and </ANSWER>."
GERMANE = "Reflect on your prior knowledge to gain a deeper understanding of the problem before solving it."
MTCOGNITION = "Self-verify your response thoroughly to ensure each reasoning step is correct."
REWARD = "You will be awarded 100 USD for every correct reasoning step."

FILE_PATH = "data/commonsenseqa_200.json"
# SAVED_PATH = "inference-codes/output/qwen/original/qwen_commonsenseqa_ans.csv"
SAVED_PATH = "inference-codes/output/qwen/formal/formal_qwen_combined_commonsenseqa_ans_ver12_dot.csv"
SAVED_PATH = "inference-codes/output/qwen/original/qwen_combined_commonsenseqa_ans_ver12_dot.csv"

CHOICES = "ABCDEFGH"

saved_data = []
with open(FILE_PATH) as file:
    data = json.load(file)
    for sample in tqdm(data):
        question = sample["question"]
        choices = sample["choices"]["text"]
        answer_string = " | ".join([CHOICES[idx] + ". " + choices[idx] for idx in range(len(choices))])
        correct_char = sample["answerKey"]
        
        pol_ger_prompt = f"Please {ZS_COT}" + f"\nQuestion: {question}\nChoices: {answer_string}." + f"\n{GERMANE}"
        mt_rw_prompt = ZS_COT + f"\nQuestion: {question}\nChoices: {answer_string}." + f"\n{MTCOGNITION}" + f"\n{REWARD}"
        pol_ger_mt_prompt = f"Please {ZS_COT}" + f"\nQuestion: {question}\nChoices: {answer_string}." + f"\n{GERMANE}" + f"\n{MTCOGNITION}"
                
        pol_ger_ans = get_qwen_answer(pol_ger_prompt)
        mt_rw_ans = get_qwen_answer(mt_rw_prompt)
        pol_ger_mt_ans = get_qwen_answer(pol_ger_mt_prompt)
        
        # print(question)
        # print("//////")
        # print(zs_ans)
        # print("###")
        # print(politeness_ans)
        # print("###")
        # print(germane_ans)
        # print("###")
        # print(mt_ans)
        # print("###")
        # print(reward_ans)
        
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
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

device = "cuda" # the device to load the model onto

access_token = "YOUR-TOKEN-HERE"
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=access_token)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=access_token)

###########################
def get_llama_answer(prompt):
    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=4096)
    assistant_message = tokenizer.batch_decode(generated_ids)[0]
    return assistant_message.split("<|end_header_id|>\n")[-1].split("<|eot_id|>")[0].strip()

ZS_COT = "Answer the following question step-by-step. Wrap your final answer choice within <ANSWER> and </ANSWER>."
GERMANE = "Reflect on your prior knowledge to gain a deeper understanding of the problem before solving it."
MTCOGNITION = "Self-verify your response thoroughly to ensure each reasoning step is correct."
REWARD = "You will be awarded 100 USD for every correct reasoning step."

FILE_PATH = "/home/ComplexPrompt/src/properties-eval/data/mmlu_200.json"
SAVED_PATH = "/home/ComplexPrompt/src/properties-eval/codes/output/llama_mmlu_ans.csv"

CHOICES = "ABCDEFGH"

saved_data = []
with open(FILE_PATH) as file:
    data = json.load(file)
    for sample in data:
        question = sample["question"]
        choices = sample["choices"]
        answer_string = " | ".join([CHOICES[idx] + ". " + choices[idx] for idx in range(len(choices))])
        correct_char = CHOICES[sample["answer"]]
        
        zs_prompt = ZS_COT + f"\nQuestion: {question}\nChoices: {answer_string}"
        politeness_prompt = f"Please {ZS_COT}" + f"\nQuestion: {question}\nChoices: {answer_string}"
        germane_prompt = ZS_COT + f"\nQuestion: {question}\nChoices: {answer_string}" + f"\n{GERMANE}"
        mt_prompt = ZS_COT + f"\nQuestion: {question}\nChoices: {answer_string}" + f"\n{MTCOGNITION}"
        reward_prompt = ZS_COT + f"\nQuestion: {question}\nChoices: {answer_string}" + f"\n{REWARD}"
                
        zs_ans = get_llama_answer(zs_prompt)
        politeness_ans = get_llama_answer(politeness_prompt)
        germane_ans = get_llama_answer(germane_prompt)
        mt_ans = get_llama_answer(mt_prompt)
        reward_ans = get_llama_answer(reward_prompt)
        
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
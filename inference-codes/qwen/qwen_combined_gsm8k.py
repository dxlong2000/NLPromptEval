import json
import csv
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto
# model_id = "finetuning-codes/output/qwen-2.5-7b-alpaca-instruct-2452025"
model_id = "/disk2/Long/WhatMakesAGoodPrompt/finetuning-codes/output/qwen-2.5-7b-alpaca-instruct-2452025-ver12"
hf_token = "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"

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def get_qwen_answer(prompt):
    # messages = [
    #     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    #     {"role": "user", "content": prompt}
    # ]
    # prompt = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generate_ids = model.generate(inputs.input_ids, max_new_tokens=1024, do_sample=True, top_p = 0.9)
    answer = tokenizer.decode(generate_ids[0][len(inputs.input_ids[0]):]).strip()
    # print(answer)
    return answer.replace("<|endoftext|>", "")

ZS_COT = "Answer the following question step-by-step. Wrap your final answer choice within <ANSWER> and </ANSWER>."
GERMANE = "Reflect on your prior knowledge to gain a deeper understanding of the problem before solving it."
MTCOGNITION = "Self-verify your response thoroughly to ensure each reasoning step is correct."
REWARD = "You will be awarded 100 USD for every correct reasoning step."

FILE_PATH = "data/gsm8k_200.json"
# SAVED_PATH = "inference-codes/output/qwen/formal/formal_qwen_combined_gmsk8_200_ans_ver12_dot.csv"
SAVED_PATH = "inference-codes/output/qwen/original/qwen_combined_gmsk8_200_ans_ver12_dot.csv"

saved_data = []
with open(FILE_PATH) as file:
    data = json.load(file)
    for sample in tqdm(data):
        question = sample["question"]
        cot_answer = sample["answer"]
        answer = sample["answer"].split("####")[1].strip()
        
        pol_ger_prompt = f"Please {ZS_COT}" + f"\nQuestion: {question}" + f"\n{GERMANE}"
        mt_rw_prompt = ZS_COT + f"\nQuestion: {question}" + f"\n{MTCOGNITION}" + f"\n{REWARD}"
        pol_ger_mt_prompt = f"Please {ZS_COT}" + f"\nQuestion: {question}" + f"\n{GERMANE}" + f"\n{MTCOGNITION}"
                
        pol_ger_ans = get_qwen_answer(pol_ger_prompt)
        mt_rw_ans = get_qwen_answer(mt_rw_prompt)
        pol_ger_mt_ans = get_qwen_answer(pol_ger_mt_prompt)
        
        saved_data.append([
            question, cot_answer, answer,
            pol_ger_ans, mt_rw_ans, pol_ger_mt_ans,
        ])

with open(SAVED_PATH, "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(["question", "cot_answer", "answer",
        "pol_ger_ans", "mt_rw_ans", "pol_ger_mt_ans",
    ])
    csvwriter.writerows(saved_data)
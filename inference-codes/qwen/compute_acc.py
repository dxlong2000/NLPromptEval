import csv 
import re
import json 

def extract_answer(text):
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
    return match.group(1) if match else "A"

def extract_answer(text):
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
    if match: return match.group(1).strip()
    match = re.search(r"ANSWER:\s*(.*)", text)
    if match: return match.group(1).strip()
    return "A"


FILE_PATH = "inference-codes/output/qwen/original/qwen_arc_c_200_ans_ver12_dot.csv"

all_cnt = 0

zs_cnt = 0
pol_cnt = 0
ger_cnt = 0
mt_cnt = 0
reward_cnt = 0

with open(FILE_PATH) as file:
    csvreader = csv.reader(file)
    header = list(next(csvreader))
    for row in csvreader:
        # question,choices,correct_char,zs_ans,politeness_ans,germane_ans,mt_ans,reward_ans
        question = row[header.index("question")]
        correct_char = row[header.index("correct_char")]
        zs_ans = row[header.index("zs_ans")]
        try: zs_ext = extract_answer(zs_ans).strip()[0]
        except: zs_ext = ""
            
        zs_cnt += (correct_char == zs_ext)
        
        pol_ans = row[header.index("politeness_ans")]
        try:
            pol_ext = extract_answer(pol_ans).strip()[0]
        except:
            pol_ext = ""
        pol_cnt += (correct_char == pol_ext)
        
        ger_ans = row[header.index("germane_ans")]
        try: 
            ger_ext = extract_answer(ger_ans).strip()[0]
        except:
            ger_ext = ""
        ger_cnt += (correct_char == ger_ext)
        
        mt_ans = row[header.index("mt_ans")]
        try:    
            mt_ext = extract_answer(mt_ans).strip()[0]
        except:
            rw_ext = ""
        mt_cnt += (correct_char == mt_ext)
        
        rw_ans = row[header.index("reward_ans")]
        try: 
            rw_ext = extract_answer(rw_ans).strip()[0]
        except:
            rw_ext = ""
        reward_cnt += (correct_char == rw_ext)
        
        all_cnt += 1

print(f"zs_cnt: {zs_cnt/all_cnt}")
print(f"pol_cnt: {pol_cnt/all_cnt}")
print(f"ger_cnt: {ger_cnt/all_cnt}")
print(f"mt_cnt: {mt_cnt/all_cnt}")
print(f"reward_cnt: {reward_cnt/all_cnt}")
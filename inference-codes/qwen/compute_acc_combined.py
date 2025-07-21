import csv 
import re
import json 

def extract_answer(text):
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
    if match: return match.group(1).strip()
    match = re.search(r"ANSWER:\s*(.*)", text)
    if match: return match.group(1).strip()
    return "A"

FILE_PATH = "inference-codes/output/qwen/original/qwen_combined_arc_c_200_ans_ver12_dot.csv"

all_cnt = 0

pol_ger_cnt = 0
mt_rw_cnt = 0
pol_ger_mt_cnt = 0

with open(FILE_PATH) as file:
    csvreader = csv.reader(file)
    header = list(next(csvreader))
    for row in csvreader:
        # question,choices,correct_char,pol_ger_ans,mt_rw_ans,pol_ger_mt_ans
        question = row[header.index("question")]
        correct_char = row[header.index("correct_char")]
        # for dt in ground_truth_data:
        #     if question == dt["question"]:

        pol_ger_ans = row[header.index("pol_ger_ans")]
        try: zs_ext = extract_answer(pol_ger_ans).strip()[0]
        except: zs_ext = ""
        pol_ger_cnt += (correct_char == zs_ext)
        
        mt_rw_ans = row[header.index("mt_rw_ans")]
        # pol_ext = extract_answer(mt_rw_ans).strip()[0]
        try: pol_ext = extract_answer(mt_rw_ans).strip()[0]
        except: pol_ext = ""
        mt_rw_cnt += (correct_char == pol_ext)
        
        pol_ger_mt_ans = row[header.index("pol_ger_mt_ans")]
        try: ger_ext = extract_answer(pol_ger_mt_ans).strip()[0]
        except: ger_ext = ""
        pol_ger_mt_cnt += (correct_char == ger_ext)
        
        all_cnt += 1

print(f"pol_ger_cnt: {pol_ger_cnt/all_cnt}")
print(f"mt_rw_cnt: {mt_rw_cnt/all_cnt}")
print(f"pol_ger_mt_cnt: {pol_ger_mt_cnt/all_cnt}")

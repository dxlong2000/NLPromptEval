import csv 
import re
import json 

def extract_gsm8k_answer(text):
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
    if match and match.group(1).strip() != "": return match.group(1)
    else: return "0"

FILE_PATH = "inference-codes/output/qwen/original/qwen_gmsk8_200_ans_ver12_dot.csv"

all_cnt = 0

zs_cnt = 0
pol_cnt = 0
ger_cnt = 0
mt_cnt = 0
reward_cnt = 0

import re

def convert_string_to_int(s: str) -> int:
    """
    Converts a string representing a number into an integer.
    The input string may include commas, currency symbols (e.g., '$'), 
    and trailing non-numeric text.
    
    Examples:
        "$7.00"     -> 7
        "7"         -> 7
        "14,000"    -> 14000
        "1128 minutes" -> 1128

    Args:
        s (str): The string to convert.
    
    Returns:
        int: The parsed integer value.
    
    Raises:
        ValueError: If no numeric value can be extracted.
    """
    # Remove common currency symbols and commas
    cleaned = s.replace("$", "").replace("%", "").replace(",", "").strip()
    
    # Use regex to find the first occurrence of a number (integer or decimal)
    match = re.search(r"[-+]?\d*\.?\d+", cleaned)
    if match:
        value = float(match.group(0))
        return int(value)
    else:
        if s == " and " or '\ntotal_area\n':
            return None
        raise ValueError(f"Cannot convert '{s}' to an integer.")


with open(FILE_PATH) as file:
    csvreader = csv.reader(file)
    header = list(next(csvreader))
    for row in csvreader:
        # question,choices,correct_char,zs_ans,politeness_ans,germane_ans,mt_ans,reward_ans
        question = row[header.index("question")]
        correct_char = convert_string_to_int(row[header.index("answer")])

        # Process zs_ans
        zs_ans = row[header.index("zs_ans")]
        zs_ext = convert_string_to_int(extract_gsm8k_answer(zs_ans))
        zs_cnt += (correct_char == zs_ext)
        if correct_char != zs_ext:
            print(f"Question: {question}\nCorrect Answer: {correct_char}\nZS Answer: {zs_ext}\n")

        # Process politeness_ans
        pol_ans = row[header.index("politeness_ans")]
        pol_ext = convert_string_to_int(extract_gsm8k_answer(pol_ans))
        pol_cnt += (correct_char == pol_ext)
        if correct_char != pol_ext:
            print(f"Question: {question}\nCorrect Answer: {correct_char}\nPoliteness Answer: {pol_ext}\n")

        # Process germane_ans
        ger_ans = row[header.index("germane_ans")]
        ger_ext = convert_string_to_int(extract_gsm8k_answer(ger_ans))
        ger_cnt += (correct_char == ger_ext)
        if correct_char != ger_ext:
            print(f"Question: {question}\nCorrect Answer: {correct_char}\nGermane Answer: {ger_ext}\n")

        # Process mt_ans
        mt_ans = row[header.index("mt_ans")]
        mt_ext = convert_string_to_int(extract_gsm8k_answer(mt_ans))
        mt_cnt += (correct_char == mt_ext)
        if correct_char != mt_ext:
            print(f"Question: {question}\nCorrect Answer: {correct_char}\nMT Answer: {mt_ext}\n")

        # Process reward_ans
        rw_ans = row[header.index("reward_ans")]
        rw_ext = convert_string_to_int(extract_gsm8k_answer(rw_ans))
        reward_cnt += (correct_char == rw_ext)
        if correct_char != rw_ext:
            print(f"Question: {question}\nCorrect Answer: {correct_char}\nReward Answer: {rw_ext}\n")

        all_cnt += 1

print(f"zs_cnt: {zs_cnt/all_cnt}")
print(f"pol_cnt: {pol_cnt/all_cnt}")
print(f"ger_cnt: {ger_cnt/all_cnt}")
print(f"mt_cnt: {mt_cnt/all_cnt}")
print(f"reward_cnt: {reward_cnt/all_cnt}")
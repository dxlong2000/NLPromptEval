import csv 
import re
import json 

def extract_gsm8k_answer(text):
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
    if match and match.group(1).strip() != "": return match.group(1)
    else: return "0"

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
        if s == " and " or s == "Ruby" or s == "Fawn" or s == 'quarters' or s == 'None' or s == 'A' or s == 'Sam' or s == 'None of the above' or s == 'Not enough information':
            return None
        raise ValueError(f"Cannot convert '{s}' to an integer.")

FILE_PATH = "inference-codes/output/qwen/original/qwen_combined_gmsk8_200_ans_ver12_dot.csv"

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
        correct_char = convert_string_to_int(row[header.index("answer")])
        # for dt in ground_truth_data:
        #     if question == dt["question"]:

        pol_ger_ans = row[header.index("pol_ger_ans")]
        zs_ext = convert_string_to_int(extract_gsm8k_answer(pol_ger_ans))
        pol_ger_cnt += (correct_char == zs_ext)
        if correct_char != zs_ext:
            print(f"Question: {question}\nCorrect Answer: {correct_char}\nPol ger Answer: {zs_ext}\n")
        
        mt_rw_ans = row[header.index("mt_rw_ans")]
        # pol_ext = extract_answer(mt_rw_ans).strip()[0]
        pol_ext = convert_string_to_int(extract_gsm8k_answer(mt_rw_ans))
        mt_rw_cnt += (correct_char == pol_ext)
        if correct_char != pol_ext:
            print(f"Question: {question}\nCorrect Answer: {correct_char}\nMt rw Answer: {zs_ext}\n")
        
        pol_ger_mt_ans = row[header.index("pol_ger_mt_ans")]
        ger_ext = convert_string_to_int(extract_gsm8k_answer(pol_ger_mt_ans))
        pol_ger_mt_cnt += (correct_char == ger_ext)
        if correct_char != ger_ext:
            print(f"Question: {question}\nCorrect Answer: {correct_char}\nPol ger mt Answer: {zs_ext}\n")
        
        all_cnt += 1

print(f"pol_ger_cnt: {pol_ger_cnt/all_cnt}")
print(f"mt_rw_cnt: {mt_rw_cnt/all_cnt}")
print(f"pol_ger_mt_cnt: {pol_ger_mt_cnt/all_cnt}")

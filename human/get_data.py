import csv 
import pandas as pd
from collections import Counter

FILE_PATH = "/home/ComplexPrompt/src/properties-eval/evaluated_property_human_eval_final.csv"

DIM_PROPERTIES = {
    "com": ['Token quantity', 'Manner', 'Interaction', 'Politeness'],
    "cog": ['Intrinsic load', 'Extraneous load', 'Germane load'],
    "ins": ['Objectives', 'External tools', 'Metacognition', 'Demos', 'Rewards'],
    "logic": ['Structural logic', 'Contextual logic'],
    "hall": ['Hallucination awareness', 'Factuality and creativity'],
    "res": ['Bias', 'Safety', 'Privacy', 'Reliability', 'Societal norms']
}

def most_frequent_number(numbers):
    if not numbers:
        return None  
    frequency = Counter(numbers)
    most_frequent = max(frequency, key=frequency.get)
    return most_frequent

def process_extracted_scores(extracted_scores):
    avg_scores = {}
    for ele in extracted_scores:
        for key in ele:
            if key not in avg_scores: avg_scores[key] = []
            avg_scores[key].append(ele[key])
    avg_return = {}
    for key in avg_scores: 
        if key == "Quantity": 
            # avg_return["Token quantity"] = sum(avg_scores[key])/len(avg_scores[key])
            avg_return["Token quantity"] = most_frequent_number(avg_scores[key])
        else: avg_return[key] = most_frequent_number(avg_scores[key])
    return avg_return

all_rows = []
with open(FILE_PATH) as file:
    csvreader = csv.reader(file)
    header = list(next(csvreader))
    # prompt,source,task,human or machine,complexity,com_eval,com_extracted_eval,cog_eval,cog_extracted_eval,ins_eval,ins_extracted_eval,logic_eval,logic_extracted_eval,hall_eval,hall_extracted_eval,res_eval,res_extracted_eval
    for row in csvreader:
        all_eval = {
            "prompt": row[0],
            "source": row[1]
        }
        for colname in header:
            if "_extracted_" in colname:
                eval_outcome = process_extracted_scores(eval(row[header.index(colname)]))
                all_eval.update(eval_outcome)
        all_rows.append(all_eval)

# import random 
# random.shuffle(all_rows)

df = pd.DataFrame(all_rows[:100])
df.to_csv('property_human_eval_final.csv', index=False)


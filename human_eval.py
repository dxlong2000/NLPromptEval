import csv 
import random 
from collections import Counter

FILE_PATH = "/home/ComplexPrompt/src/properties-eval/evaluated_test_data.csv"
saved_data = []
with open(FILE_PATH) as file:
    csvreader = csv.reader(file)
    header = list(next(csvreader))
    # prompt,source,task,human or machine,complexity,com_eval,com_extracted_eval,cog_eval,cog_extracted_eval,ins_eval,ins_extracted_eval,logic_eval,logic_extracted_eval,hall_eval,hall_extracted_eval,res_eval,res_extracted_eval
    for row in csvreader:
        prompt = row[header.index("prompt")]
        source = row[header.index("source")]
        saved_data.append([
            prompt, source
        ])

header = [
    'prompt', 'source', 'Token quantity', 'Manner', 'Interaction', 
    'Politeness', 'Intrinsic load', 'Extraneous load', 
    'Germane load', 'Objectives', 'External tools', 
    'Metacognition', 'Demos', 'Rewards', 'Structural logic', 
    'Contextual logic', 'Hallucination awareness', 'Factuality and creativity',
    'Bias', 'Safety', 'Privacy', 'Reliability', 'Societal norms'
]

random.shuffle(saved_data)
with open("/home/ComplexPrompt/src/properties-eval/property_human_eval.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(header)
    csvwriter.writerows(saved_data[:100])


import csv 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

FILE_PATH2 = "/home/WhatMakesAGoodPrompt/evaluated_test_data_multiturn_gemini2flash.csv"
FILE_PATH = "/home/WhatMakesAGoodPrompt/evaluated_test_data_gemini2flash.csv"

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

num_rows = 0
properties_scores = {}
for PATH in [FILE_PATH, FILE_PATH2]:
    with open(PATH) as file:
        csvreader = csv.reader(file)
        header = list(next(csvreader))
        for row in csvreader:
            num_rows += 1
            for name in header:
                if "_extracted_" in name: 
                    extracted_scores = eval(row[header.index(name)])
                    processed_extracted_scores = process_extracted_scores(extracted_scores)
                    for key in processed_extracted_scores:
                        if key not in properties_scores: properties_scores[key] = []
                        properties_scores[key].append(processed_extracted_scores[key])

print(f"num_rows: {num_rows}")
print("===")
for property in properties_scores:
    print(f"{property}: {sum(properties_scores[property])/len(properties_scores[property])}")
    print(len(properties_scores[property]))

df = pd.DataFrame(properties_scores)
correlation_matrix = df.corr()

    
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

avg_threshold = 5
avg_scores = df.mean()
low_score_mask = (avg_scores.values[:, None] < avg_threshold) & (avg_scores.values[None, :] < avg_threshold)

plt.figure(figsize=(15, 5))
ax = sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cbar_kws={'pad': 0.01},
    cmap="coolwarm",
    # cmap=sns.light_palette("red", as_cmap=True),
    cbar=True
)

# Add hatching for pairs where both average scores are less than 0.5
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        if low_score_mask[i, j] and not mask[i, j]:
            plt.gca().add_patch(
                plt.Rectangle(
                    (j, i),  # (x, y) coordinates of the rectangle
                    1, 1,  # Width and height of the rectangle
                    fill=False,  # No fill
                    hatch="\\\\",  # Hatching pattern
                    edgecolor="black",  # Edge color of the hatching
                    lw=0.5  # Line width
                )
            )

# Add title and labels
plt.xticks(rotation=20, ha="right")
plt.yticks(rotation=0)

# Save as PDF
plt.tight_layout()
plt.savefig("correlation_heatmap_gemini2flash.pdf")
plt.close()

# Find pairs with correlation > 0.7
threshold = 0.7
high_correlation_pairs = []

cntcnt = 0
for i in range(len(correlation_matrix.columns)):
    for j in range(i):  # Only check the lower triangle
        cntcnt += 1
        if correlation_matrix.iloc[i, j] > threshold:
            high_correlation_pairs.append(
                (correlation_matrix.index[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j])
            )

# Output the pairs
print("===")
print(f"Pairs with correlation more than {threshold}: {len(high_correlation_pairs)}")
print(f"cntcnt: {cntcnt}")
print("===")
for pair in high_correlation_pairs:
    print(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")
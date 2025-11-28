import pandas as pd
import csv
import random



input_file="data/KDDTrain_balanced.txt"
output_file="data/KDDTrain+_balanced_to_poisoned_DOS 15%.txt"

df = pd.read_csv(input_file, header=None)

column_index  = 41
target_value = "normal"


normal = (df[column_index] == target_value).sum()


attack = (df[column_index] != target_value).sum()

total = normal + attack
print("\nIn the original train data set the Total count is: ", total)
print(f"Count of normals: {normal} Which is approximately: {normal/total*100:.2f} %")
print(f"Count of attacks: {attack} Which is approximately: {attack/total*100:.2f} %\n")

print(f"--------------------------------------------------------------")

print(f"\n\nThe posioned data set counts\n")

target_column=41

original_value = "normal"
value_poison = "neptune"
flip_count_goal = 0.15 * total

flip_count = 0
flip_probability = 0.45
count_val1 = 0
row_index = 0

with open(input_file, newline='', encoding='utf-8') as infile, \
    open(output_file, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if row[41] == original_value:
            if flip_count < flip_count_goal and random.random() < flip_probability:
                row[41] = value_poison
                flip_count += 1
        writer.writerow(row)

df = pd.read_csv(output_file, header=None)

column_index  = 41
target_value = "normal"


normal = (df[column_index] == target_value).sum()


attack = (df[column_index] != target_value).sum()

total = normal + attack
print("\nIn the POISONED train data set the Total count is: ", total, " and the total flipped is: ", flip_count)
print(f"Count of POISONED normals: {normal} Which is approximately: {normal/total*100:.2f} %")
print(f"Count of POISONED attacks: {attack} Which is approximately: {attack/total*100:.2f} %\n")
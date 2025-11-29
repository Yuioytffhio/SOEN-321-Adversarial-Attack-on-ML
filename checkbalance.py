import pandas as pd

df = pd.read_csv("data/KDDtrain+_balanced_to_poisoned_randomflipping 20%.txt", header=None)

column_index  = 41
target_value = "normal"


normal = (df[column_index] == target_value).sum()


attack = (df[column_index] != target_value).sum()

total = normal + attack
print("\nIn the test data set the Total count is: ", total)
print(f"Count of normals: {normal} Which is approximately: {normal/total*100:.2f} %")
print(f"Count of attacks: {attack} Which is approximately: {attack/total*100:.2f} %\n")

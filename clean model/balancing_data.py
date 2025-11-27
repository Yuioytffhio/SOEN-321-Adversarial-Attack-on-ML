import pandas as pd

# Load dataset
df = pd.read_csv("data/KDDTrain+.txt", header=None)

# Label column
column_index = 41
target_value = "normal"

# Count normal and attack samples
normal = (df[column_index] == target_value).sum()
attack = (df[column_index] != target_value).sum()
total = normal + attack

# Print dataset distribution
print("\nIn the train dataset the Total count is:", total)
print(f"Count of normals: {normal}  ({normal/total*100:.2f}%)")
print(f"Count of attacks: {attack}  ({attack/total*100:.2f}%)\n")



print("Balancing dataset to a 50/50 ratio...")

# Separate normal and attack rows
df_normal = df[df[column_index] == target_value]
df_attack = df[df[column_index] != target_value]

# Choose the minority size
min_size = min(len(df_normal), len(df_attack))

# Downsample both classes to the same size
df_normal_bal = df_normal.sample(min_size, random_state=42)
df_attack_bal = df_attack.sample(min_size, random_state=42)

# Combine balanced dataset
df_balanced = pd.concat([df_normal_bal, df_attack_bal]).sample(frac=1, random_state=42)

# Save balanced dataset
df_balanced.to_csv("data/KDDTrain_balanced.txt", index=False, header=False)

print("\nBalanced dataset created!")
print("New size:", len(df_balanced))
print(f"Normals: {len(df_normal_bal)}")
print(f"Attacks: {len(df_attack_bal)}")
print("\nSaved as: data/KDDTrain_balanced.txt")

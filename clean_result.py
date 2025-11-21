import glob
import re

# Step 1: Load all .txt files
filename = "logs/results_bagsize50_mixed_t2_9_12_2025.txt"  # change if files are in a subfolder

# Step 1: Open with correct encoding
with open(filename, 'r', encoding='utf-16') as f:
    content = f.read()

# Step 2: Remove all whitespace
cleaned = ''.join(content.split())  # removes spaces, newlines, tabs
#print("🧹 Cleaned content preview:", cleaned[:100])  # optional preview

# Step 3: Regex match: best_test followed by a float
matches = re.findall(r'finaltest=([0-9]*\.?[0-9]+)', cleaned)
values = [round(float(m)*100,2) for m in matches]
#print("✅ Extracted values:", values)

# Step 4: Group into 10s
grouped = [values[i:i+10] for i in range(0, len(values), 10)]
for i, group in enumerate(grouped):
    print(f"Group {i+1}: {group}")
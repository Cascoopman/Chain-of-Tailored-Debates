from helpers import *
from datasets import load_dataset

# Retrieve the HaluEval dataset from Hugginface
ds = load_dataset("pminervini/HaluEval", "summarization")
dataset = ds['data']
df = dataset.to_pandas()

length_document = 0
length_right_summary = 0
length_hallucinated_summary = 0

print(len(df))
print(df.columns)   

count = 0

# Iterate over all rows
for i in range(len(df)):
    row = df.iloc[i]
    document = row['document']
    right_summary = row['right_summary']
    hallucinated_summary = row['hallucinated_summary']

    if "CLICK HERE" in right_summary or "CLICK HERE" in hallucinated_summary or "CLICK HERE" in document:
        continue
    
    if len(hallucinated_summary) > (1.05 * len(right_summary)) or len(hallucinated_summary) < (0.95 * len(right_summary)):
        continue
    
    count += 1
        
    print("Right Summary:", right_summary, "\n")
    print("Hallucinated Summary:", hallucinated_summary,"\n")

print(count)
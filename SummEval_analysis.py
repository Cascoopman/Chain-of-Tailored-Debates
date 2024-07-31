from helpers import *
from datasets import load_dataset

# Retrieve the SummEval dataset from Hugginface
ds = load_dataset("mteb/summeval")
dataset = ds['test']
df = dataset.to_pandas()

length_document = 0
length_right_summary = 0
length_hallucinated_summary = 0

print(len(df))
print(df.columns)   

count = 0
difference = 0

# Iterate over all rows
for i in range(len(df)):
    row = df.iloc[i]
    document = row['text']
    right_summary = find_random_summary_with_consistency_5(row)
    hallucinated_summary = find_random_summary_with_hallucinations(row)
        
    if right_summary == None or hallucinated_summary == None:
        continue
    
    #if len(hallucinated_summary) > (1.15 * len(right_summary)) or len(hallucinated_summary) < (0.85 * len(right_summary)):
    #    continue
    
    length_document += len(document)
    length_right_summary += len(right_summary)
    length_hallucinated_summary += len(hallucinated_summary)
    difference += abs(len(right_summary) - len(hallucinated_summary))
    
    count += 1
    print("-" * 100)
    print("Document:", document, "\n")
    print("Right Summary:", right_summary, "\n")
    print("Hallucinated Summary:", hallucinated_summary,"\n")

print("Average length of document:", length_document / len(df))
print("Average length of right summary:", length_right_summary / len(df))
print("Average length of hallucinated summary:", length_hallucinated_summary / len(df))
print("Average difference between right and hallucinated summary:", difference / count)
print(count)
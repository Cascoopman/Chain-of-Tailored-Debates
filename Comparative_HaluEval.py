from helpers import *
from datasets import load_dataset
from sklearn.metrics import f1_score
import pandas as pd
from datetime import datetime

# Retrieve the HaluEval dataset from Hugginface
ds = load_dataset("pminervini/HaluEval", "summarization")
dataset = ds['data']
df = dataset.to_pandas()

# Remove the first three rows used for fewshot prompting
df = df.iloc[3:]

# Scramble the dataset
df = df.sample(frac=1, random_state=19).reset_index(drop=True)

# TODO: Set parameters for the number of rows to analyse
n = 25
count = 0

# Initialize lists to store true labels and predicted labels for F1 score calculation
true_labels = []
baseline_gpt35_preds = []
baseline_gpt4_preds = []
baseline_gpt4o_mini_preds = []
sentence_gpt35_preds = []
sentence_gpt4_preds = []
sentence_gpt4o_mini_preds = []

for i in range(len(df)):
    
    if count == n:
        break
    
    row = df.iloc[i]
    document = row['document']
    right_summary = row['right_summary']
    hallucinated_summary = row['hallucinated_summary']
    
    results_per_iteration = pd.DataFrame(columns=[
        'Row', 
        'Document', 
        'Right Summary', 
        'Hallucinated Summary', 
        'True Label', 
        'Baseline GPT-3.5 Prediction', 
        'Baseline GPT-4 Prediction',
        'Baseline GPT-4o Mini Prediction',
        'Sentence Level GPT-3.5 Prediction',
        'Sentence Level GPT-4 Prediction',
        'Sentence Level GPT-4o Mini Prediction'
    ])

    print("-" * 100)
    print("ANALYZING ROW", i+1, "OF", n, "...")
    print("-" * 100)

    # Print the values of the selected row
    print("Document:", document)
    print("Right Summary:", right_summary)
    print("Hallucinated Summary:", hallucinated_summary)
    print("-" * 100)
    
    # Run the analysis for right summary
    print("Running the process for a non-hallucinated summary")
    true_labels.append(0)
    
    baseline_gpt35_pred = baseline(document, right_summary, "gpt35")
    baseline_gpt35_preds.append(baseline_gpt35_pred)
    
    baseline_gpt4_pred = baseline(document, right_summary, "gpt4")
    baseline_gpt4_preds.append(baseline_gpt4_pred)
    
    baseline_gpt4o_mini_pred = baseline(document, right_summary, "gpt4o_mini")
    baseline_gpt4o_mini_preds.append(baseline_gpt4o_mini_pred)
    
    sentence_gpt35_pred = sentence_level("gpt35", document, right_summary)
    sentence_gpt35_preds.append(sentence_gpt35_pred)
    
    sentence_gpt4_pred = sentence_level("gpt4", document, right_summary)
    sentence_gpt4_preds.append(sentence_gpt4_pred)
    
    sentence_gpt4o_mini_pred = sentence_level("gpt4o_mini", document, right_summary)
    sentence_gpt4o_mini_preds.append(sentence_gpt4o_mini_pred)
       
    # Save the results of the current iteration for the true summary
    results_per_iteration = pd.concat([results_per_iteration, pd.DataFrame([{
        'Row': i+1, 
        'Document': document, 
        'Right Summary': right_summary, 
        'Hallucinated Summary': '',
        'True Label': 0,
        'Baseline GPT-3.5 Prediction': baseline_gpt35_pred,
        'Baseline GPT-4 Prediction': baseline_gpt4_pred,
        'Baseline GPT-4o Mini Prediction': baseline_gpt4o_mini_pred,
        'Sentence Level GPT-3.5 Prediction': sentence_gpt35_pred,
        'Sentence Level GPT-4 Prediction': sentence_gpt4_pred,
        'Sentence Level GPT-4o Mini Prediction': sentence_gpt4o_mini_pred,
    }])], ignore_index=True)
    
    # Run the analysis for hallucinated summary
    print("-" * 100)
    print("Running the process for a hallucinated summary")
    true_labels.append(1) 
    
    baseline_gpt35_pred = baseline(document, hallucinated_summary, "gpt35")
    baseline_gpt35_preds.append(baseline_gpt35_pred)
    
    baseline_gpt4_pred = baseline(document, hallucinated_summary, "gpt4")
    baseline_gpt4_preds.append(baseline_gpt4_pred)
    
    baseline_gpt4o_mini_pred = baseline(document, hallucinated_summary, "gpt4o_mini")
    baseline_gpt4o_mini_preds.append(baseline_gpt4o_mini_pred)
    
    sentence_gpt35_pred = sentence_level("gpt35", document, hallucinated_summary)
    sentence_gpt35_preds.append(sentence_gpt35_pred)
    
    sentence_gpt4_pred = sentence_level("gpt4", document, hallucinated_summary)
    sentence_gpt4_preds.append(sentence_gpt4_pred)
    
    sentence_gpt4o_mini_pred = sentence_level("gpt4o_mini", document, hallucinated_summary)
    sentence_gpt4o_mini_preds.append(sentence_gpt4o_mini_pred)
    
    # Save the results of the current iteration for the hallucinated summary
    results_per_iteration = pd.concat([results_per_iteration, pd.DataFrame([{
        'Row': i+1, 
        'Document': document, 
        'Right Summary': '',
        'Hallucinated Summary': hallucinated_summary,
        'True Label': 1,
        'Baseline GPT-3.5 Prediction': baseline_gpt35_pred,
        'Baseline GPT-4 Prediction': baseline_gpt4_pred,
        'Baseline GPT-4o Mini Prediction': baseline_gpt4o_mini_pred,
        'Sentence Level GPT-3.5 Prediction': sentence_gpt35_pred,
        'Sentence Level GPT-4 Prediction': sentence_gpt4_pred,
        'Sentence Level GPT-4o Mini Prediction': sentence_gpt4o_mini_pred,
    }])], ignore_index=True)
    count += 1
    # Save the current iteration results to a CSV file
    results_per_iteration.to_csv("Comparative_analysis.csv", mode='a', header=not pd.io.common.file_exists("Comparative_analysis.csv"), index=False)


print("-" * 100)
print(f"Results after {n} iterations:\n")

accuracies = {}

accuracies['baseline_GPT35_TNR'] = sum([1 for i in range(0, 2*n, 2) if baseline_gpt35_preds[i] == 0]) / n
accuracies['baseline_GPT35_TPR'] = sum([1 for i in range(1, 2*n, 2) if baseline_gpt35_preds[i] == 1]) / n
accuracies['baseline_GPT4_TNR'] = sum([1 for i in range(0, 2*n, 2) if baseline_gpt4_preds[i] == 0]) / n
accuracies['baseline_GPT4_TPR'] = sum([1 for i in range(1, 2*n, 2) if baseline_gpt4_preds[i] == 1]) / n
accuracies['baseline_GPT4o_Mini_TNR'] = sum([1 for i in range(0, 2*n, 2) if baseline_gpt4o_mini_preds[i] == 0]) / n
accuracies['baseline_GPT4o_Mini_TPR'] = sum([1 for i in range(1, 2*n, 2) if baseline_gpt4o_mini_preds[i] == 1]) / n

accuracies['sentence_GPT35_TNR'] = sum([1 for i in range(0, 2*n, 2) if sentence_gpt35_preds[i] == 0]) / n
accuracies['sentence_GPT35_TPR'] = sum([1 for i in range(1, 2*n, 2) if sentence_gpt35_preds[i] == 1]) / n
accuracies['sentence_GPT4_TNR'] = sum([1 for i in range(0, 2*n, 2) if sentence_gpt4_preds[i] == 0]) / n
accuracies['sentence_GPT4_TPR'] = sum([1 for i in range(1, 2*n, 2) if sentence_gpt4_preds[i] == 1]) / n
accuracies['sentence_GPT4o_Mini_TNR'] = sum([1 for i in range(0, 2*n, 2) if sentence_gpt4o_mini_preds[i] == 0]) / n
accuracies['sentence_GPT4o_Mini_TPR'] = sum([1 for i in range(1, 2*n, 2) if sentence_gpt4o_mini_preds[i] == 1]) / n


print("Baseline GPT-3.5 TNR:", accuracies['baseline_GPT35_TNR'])
print("Baseline GPT-3.5 TPR:", accuracies['baseline_GPT35_TPR'])
print("Baseline GPT-4 TNR:", accuracies['baseline_GPT4_TNR'])
print("Baseline GPT-4 TPR:", accuracies['baseline_GPT4_TPR'])
print("Baseline GPT-4o Mini TNR:", accuracies['baseline_GPT4o_Mini_TNR'])
print("Baseline GPT-4o Mini TPR:", accuracies['baseline_GPT4o_Mini_TPR'])

print("Sentence Level GPT-3.5 TNR:", accuracies['sentence_GPT35_TNR'])
print("Sentence Level GPT-3.5 TPR:", accuracies['sentence_GPT35_TPR'])
print("Sentence Level GPT-4 TNR:", accuracies['sentence_GPT4_TNR'])
print("Sentence Level GPT-4 TPR:", accuracies['sentence_GPT4_TPR'])
print("Sentence Level GPT-4o Mini TNR:", accuracies['sentence_GPT4o_Mini_TNR'])
print("Sentence Level GPT-4o Mini TPR:", accuracies['sentence_GPT4o_Mini_TPR'])

# Calculate and print the F1 scores
baseline_gpt35_f1 = f1_score(true_labels, baseline_gpt35_preds)
baseline_gpt4_f1 = f1_score(true_labels, baseline_gpt4_preds)
baseline_gpt4o_mini_f1 = f1_score(true_labels, baseline_gpt4o_mini_preds)
sentence_gpt35_f1 = f1_score(true_labels, sentence_gpt35_preds)
sentence_gpt4_f1 = f1_score(true_labels, sentence_gpt4_preds)
sentence_gpt4o_mini_f1 = f1_score(true_labels, sentence_gpt4o_mini_preds)

print("Baseline GPT-3.5 F1 Score:", baseline_gpt35_f1)
print("Baseline GPT-4 F1 Score:", baseline_gpt4_f1)
print("Baseline GPT-4o Mini F1 Score:", baseline_gpt4o_mini_f1)
print("Sentence Level GPT-3.5 F1 Score:", sentence_gpt35_f1)
print("Sentence Level GPT-4 F1 Score:", sentence_gpt4_f1)
print("Sentence Level GPT-4o Mini F1 Score:", sentence_gpt4o_mini_f1)

# Create a DataFrame for the new results
results = pd.DataFrame({
    'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    'Number of Rows': [n],
    'Baseline GPT3.5 (F1 Score)': [baseline_gpt35_f1],
    'Baseline GPT4 (F1 Score)': [baseline_gpt4_f1],
    'Baseline GPT4o Mini (F1 Score)': [baseline_gpt4o_mini_f1],
    'Sentence Level GPT3.5 (F1 Score)': [sentence_gpt35_f1],
    'Sentence Level GPT4 (F1 Score)': [sentence_gpt4_f1],
    'Sentence Level GPT4o Mini (F1 Score)': [sentence_gpt4o_mini_f1],
    'Baseline GPT3.5 (TNR)': [accuracies['baseline_GPT35_TNR']],
    'Baseline GPT3.5 (TPR)': [accuracies['baseline_GPT35_TPR']],
    'Baseline GPT4 (TNR)': [accuracies['baseline_GPT4_TNR']],
    'Baseline GPT4 (TPR)': [accuracies['baseline_GPT4_TPR']],
    'Baseline GPT4o Mini (TNR)': [accuracies['baseline_GPT4o_Mini_TNR']],
    'Baseline GPT4o Mini (TPR)': [accuracies['baseline_GPT4o_Mini_TPR']],
    'Sentence Level GPT3.5 (TNR)': [accuracies['sentence_GPT35_TNR']],
    'Sentence Level GPT3.5 (TPR)': [accuracies['sentence_GPT35_TPR']],
    'Sentence Level GPT4 (TNR)': [accuracies['sentence_GPT4_TNR']],
    'Sentence Level GPT4 (TPR)': [accuracies['sentence_GPT4_TPR']],
    'Sentence Level GPT4o Mini (TNR)': [accuracies['sentence_GPT4o_Mini_TNR']],
    'Sentence Level GPT4o Mini (TPR)': [accuracies['sentence_GPT4o_Mini_TPR']]
    })

# Append the new results to the existing CSV file
results.to_csv("Comparative_analysis_results.csv", mode='a', header=not pd.io.common.file_exists("Comparative_analysis_results.csv"), index=False)
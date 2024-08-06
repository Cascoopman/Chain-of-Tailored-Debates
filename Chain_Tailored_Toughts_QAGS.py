from helpers import *
from datasets import load_dataset
from sklearn.metrics import f1_score
import pandas as pd
from datetime import datetime

# Read csv file
df_correct = pd.read_csv("Code/Data/QAGS/correct.csv")

df_hallucinated = pd.read_csv("Code/Data/QAGS/hallucinated.csv")

# Scramble the dataset
df_correct = df_correct.sample(frac=1, random_state=42).reset_index(drop=True)
df_hallucinated = df_hallucinated.sample(frac=1, random_state=42).reset_index(drop=True)

print(df_correct.__len__)
print(df_hallucinated.__len__)

# TODO: Set parameters for the number of rows to analyse
n = 60

# Initialize lists to store true labels and predicted labels for F1 score calculation
true_labels = []
baseline_preds = []
gpt4o_mini_sentence_level_preds = []
gpt4o_mini_statement_level_preds = []
gpt4o_mini_cot_preds = []
gpt4o_mini_cott_sentence_level_preds = []
gpt4o_mini_cott_statement_level_preds = []
count = 0

for i in range(len(df_hallucinated)):
    
    if count == n:
        break
    
    row = df_correct.iloc[i]
    document = row['article']
    right_summary = row['summary']
    
    results_per_iteration = pd.DataFrame(columns=[
        'Row', 
        'Document', 
        'Right Summary', 
        'Hallucinated Summary', 
        'True Label', 
        'Baseline Prediction', 
        'GPT4o Mini Sentence Level Prediction',
        'GPT4o Mini Statement Level Prediction', 
        'GPT4o Mini CoT Prediction',
        'GPT4o Mini CoT Reasoning',
        'GPT4o Mini CoTT Sentence Level Prediction',
        'GPT4o Mini CoTT Sentence Level Reasoning',
        'GPT4o Mini CoTT Statement Level Prediction',
        'GPT4o Mini CoTT Statement Level Reasoning',
    ])

    print("-" * 100)
    print("ANALYZING ROW", count+1, "OF", n, "...")
    print("-" * 100)

    # Print the values of the selected row
    print("Document:", document)
    print("summary:", right_summary)
    #print("Hallucinated Summary:", hallucinated_summary)
    print("-" * 100)
    
    # Run the analysis for right summary
    text = "Running the process for a non-hallucinated summary"
    print(text.upper())
    true_labels.append(0)
    
    baseline_pred = baseline(document, right_summary, 'gpt4o_mini')
    baseline_preds.append(baseline_pred)
        
    gpt4o_mini_sentence_level_pred = sentence_level("gpt4o_mini", document, right_summary)   
    gpt4o_mini_sentence_level_preds.append(gpt4o_mini_sentence_level_pred)
    
    gpt4o_mini_statement_level_pred = statement_level("gpt4o_mini", document, right_summary)   
    gpt4o_mini_statement_level_preds.append(gpt4o_mini_statement_level_pred)
                
    (gpt4o_mini_cot_pred, gpt4o_mini_cot_reasoning) = chain_thoughts(document, right_summary)
    gpt4o_mini_cot_preds.append(gpt4o_mini_cot_pred)

    (gpt4o_mini_cott_sentence_level_pred, gpt4o_mini_cott_sentence_level_reasoning) = chain_tailored_thoughts_sentence("gpt4o_mini", document, right_summary)
    gpt4o_mini_cott_sentence_level_preds.append(gpt4o_mini_cott_sentence_level_pred)
    
    (gpt4o_mini_cott_statement_level_pred, gpt4o_mini_cott_statement_level_reasoning) = chain_tailored_thoughts("gpt4o_mini", document, right_summary)
    gpt4o_mini_cott_statement_level_preds.append(gpt4o_mini_cott_statement_level_pred)
    
    
    # Save the results of the current iteration for the true summary
    results_per_iteration = pd.concat([results_per_iteration, pd.DataFrame([{
        'Row': i+1, 
        'Document': document, 
        'Right Summary': right_summary, 
        'Hallucinated Summary': '',
        'True Label': 0,
        'Baseline Prediction': baseline_pred,
        'GPT4o Mini Sentence Level Prediction': gpt4o_mini_sentence_level_pred,
        'GPT4o Mini Statement Level Prediction': gpt4o_mini_statement_level_pred,
        'GPT4o Mini CoT Prediction': gpt4o_mini_cot_pred,
        'GPT4o Mini CoT Reasoning': gpt4o_mini_cot_reasoning,
        'GPT4o Mini CoTT Sentence Level Prediction': gpt4o_mini_cott_sentence_level_pred,
        'GPT4o Mini CoTT Sentence Level Reasoning': gpt4o_mini_cott_sentence_level_reasoning,
        'GPT4o Mini CoTT Statement Level Prediction': gpt4o_mini_cott_statement_level_pred,
        'GPT4o Mini CoTT Statement Level Reasoning': gpt4o_mini_cott_statement_level_reasoning,
    }])], ignore_index=True)
    
    # Run the analysis for hallucinated summary
    
    row = df_hallucinated.iloc[i]
    document = row['article']
    hallucinated_summary = row['summary']
    
    print("-" * 100)
    text = "Running the process for a hallucinated summary"
    print(text.upper())
    true_labels.append(1) 
    
    baseline_pred = baseline(document, hallucinated_summary, 'gpt4o_mini')
    baseline_preds.append(baseline_pred)
    
    gpt4o_mini_sentence_level_pred = sentence_level("gpt4o_mini", document, hallucinated_summary)
    gpt4o_mini_sentence_level_preds.append(gpt4o_mini_sentence_level_pred)
    
    gpt4o_mini_statement_level_pred = statement_level("gpt4o_mini", document, hallucinated_summary)
    gpt4o_mini_statement_level_preds.append(gpt4o_mini_statement_level_pred)
    
    (gpt4o_mini_cot_pred, gpt4o_mini_cot_reasoning) = chain_thoughts(document, hallucinated_summary)
    gpt4o_mini_cot_preds.append(gpt4o_mini_cot_pred)
    
    (gpt4o_mini_cott_sentence_level_pred, gpt4o_mini_cott_sentence_level_reasoning) = chain_tailored_thoughts_sentence("gpt4o_mini", document, hallucinated_summary)
    gpt4o_mini_cott_sentence_level_preds.append(gpt4o_mini_cott_sentence_level_pred)
    
    (gpt4o_mini_cott_statement_level_pred, gpt4o_mini_cott_statement_level_reasoning) = chain_tailored_thoughts("gpt4o_mini", document, hallucinated_summary)
    gpt4o_mini_cott_statement_level_preds.append(gpt4o_mini_cott_statement_level_pred)
    
    
    # Save the results of the current iteration for the hallucinated summary
    results_per_iteration = pd.concat([results_per_iteration, pd.DataFrame([{
        'Row': i+1, 
        'Document': document, 
        'Right Summary': '',
        'Hallucinated Summary': hallucinated_summary,
        'True Label': 1,
        'Baseline Prediction': baseline_pred,
        'GPT4o Mini Sentence Level Prediction': gpt4o_mini_sentence_level_pred,
        'GPT4o Mini Statement Level Prediction': gpt4o_mini_statement_level_pred,
        'GPT4o Mini CoT Prediction': gpt4o_mini_cot_pred,
        'GPT4o Mini CoT Reasoning': gpt4o_mini_cot_reasoning,
        'GPT4o Mini CoTT Sentence Level Prediction': gpt4o_mini_cott_sentence_level_pred,
        'GPT4o Mini CoTT Sentence Level Reasoning': gpt4o_mini_cott_sentence_level_reasoning,
        'GPT4o Mini CoTT Statement Level Prediction': gpt4o_mini_cott_statement_level_pred,
        'GPT4o Mini CoTT Statement Level Reasoning': gpt4o_mini_cott_statement_level_reasoning,
    }])], ignore_index=True)
    
    count += 1
    
    # Save the current iteration results to a CSV file
    results_per_iteration.to_csv("chain_tailored_thoughts.csv", mode='a', header=not pd.io.common.file_exists("chain_tailored_thoughts.csv"), index=False)


print("-" * 100)
print(f"Results after {n} iterations:\n")

# Calculate and print the accuracies
accuracies = {}

accuracies['baseline_TNR'] = sum([1 for i in range(0, 2*n, 2) if baseline_preds[i] == 0]) / n
accuracies['baseline_TPR'] = sum([1 for i in range(1, 2*n, 2) if baseline_preds[i] == 1]) / n

accuracies['gpt4o_mini_sentence_level_TNR'] = sum([1 for i in range(0, 2*n, 2) if gpt4o_mini_sentence_level_preds[i] == 0]) / n
accuracies['gpt4o_mini_sentence_level_TPR'] = sum([1 for i in range(1, 2*n, 2) if gpt4o_mini_sentence_level_preds[i] == 1]) / n

accuracies['gpt4o_mini_statement_level_TNR'] = sum([1 for i in range(0, 2*n, 2) if gpt4o_mini_statement_level_preds[i] == 0]) / n
accuracies['gpt4o_mini_statement_level_TPR'] = sum([1 for i in range(1, 2*n, 2) if gpt4o_mini_statement_level_preds[i] == 1]) / n

accuracies['gpt4o_mini_cot_TNR'] = sum([1 for i in range(0, 2*n, 2) if gpt4o_mini_cot_preds[i] == 0]) / n
accuracies['gpt4o_mini_cot_TPR'] = sum([1 for i in range(1, 2*n, 2) if gpt4o_mini_cot_preds[i] == 1]) / n

accuracies['gpt4o_mini_cott_sentence_level_TNR'] = sum([1 for i in range(0, 2*n, 2) if gpt4o_mini_cott_sentence_level_preds[i] == 0]) / n
accuracies['gpt4o_mini_cott_sentence_level_TPR'] = sum([1 for i in range(1, 2*n, 2) if gpt4o_mini_cott_sentence_level_preds[i] == 1]) / n

accuracies['gpt4o_mini_cott_statement_level_TNR'] = sum([1 for i in range(0, 2*n, 2) if gpt4o_mini_cott_statement_level_preds[i] == 0]) / n
accuracies['gpt4o_mini_cott_statement_level_TPR'] = sum([1 for i in range(1, 2*n, 2) if gpt4o_mini_cott_statement_level_preds[i] == 1]) / n

print("Baseline (TNR):", accuracies['baseline_TNR'])
print("Baseline (TPR):", accuracies['baseline_TPR'])

print("GPT4o Mini Sentence Level (TNR):", accuracies['gpt4o_mini_sentence_level_TNR'])
print("GPT4o Mini Sentence Level (TPR):", accuracies['gpt4o_mini_sentence_level_TPR'])

print("GPT4o Mini Statement Level (TNR):", accuracies['gpt4o_mini_statement_level_TNR'])
print("GPT4o Mini Statement Level (TPR):", accuracies['gpt4o_mini_statement_level_TPR'])

print("GPT4o Mini CoT (TNR):", accuracies['gpt4o_mini_cot_TNR'])
print("GPT4o Mini CoT (TPR):", accuracies['gpt4o_mini_cot_TPR'])

print("GPT4o Mini CoTT Sentence Level (TNR):", accuracies['gpt4o_mini_cott_sentence_level_TNR'])
print("GPT4o Mini CoTT Sentence Level (TPR):", accuracies['gpt4o_mini_cott_sentence_level_TPR'])

print("GPT4o Mini CoTT Statement Level (TNR):", accuracies['gpt4o_mini_cott_statement_level_TNR'])
print("GPT4o Mini CoTT Statement Level (TPR):", accuracies['gpt4o_mini_cott_statement_level_TPR'])


# Calculate and print the F1 scores
baseline_f1 = f1_score(true_labels, baseline_preds)
gpt4o_mini_sentence_level_f1 = f1_score(true_labels, gpt4o_mini_sentence_level_preds)
gpt4o_mini_statement_level_f1 = f1_score(true_labels, gpt4o_mini_statement_level_preds)
gpt4o_mini_cot_f1 = f1_score(true_labels, gpt4o_mini_cot_preds)
gpt4o_mini_cott_sentence_level_f1 = f1_score(true_labels, gpt4o_mini_cott_sentence_level_preds)
gpt4o_mini_cott_statement_level_f1 = f1_score(true_labels, gpt4o_mini_cott_statement_level_preds)


print("Baseline F1 Score:", baseline_f1)
print("GPT4o Mini Sentence Level F1 Score:", gpt4o_mini_sentence_level_f1)
print("GPT4o Mini Statement Level F1 Score:", gpt4o_mini_statement_level_f1)
print("GPT4o Mini CoT F1 Score:", gpt4o_mini_cot_f1)
print("GPT4o Mini CoTT Sentence Level F1 Score:", gpt4o_mini_cott_sentence_level_f1)
print("GPT4o Mini CoTT Statement Level F1 Score:", gpt4o_mini_cott_statement_level_f1)


# Create a DataFrame for the new results
results = pd.DataFrame({
    'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    'Number of Rows': [n],
    'Baseline F1 Score': [baseline_f1],
    'GPT4o Mini Sentence Level F1 Score': [gpt4o_mini_sentence_level_f1],
    'GPT4o Mini Statement Level F1 Score': [gpt4o_mini_statement_level_f1],
    'GPT4o Mini CoT F1 Score': [gpt4o_mini_cot_f1],
    'GPT4o Mini CoTT Sentence Level F1 Score': [gpt4o_mini_cott_sentence_level_f1],
    'GPT4o Mini CoTT Statement Level F1 Score': [gpt4o_mini_cott_statement_level_f1],
})

# Append the new results to the existing CSV file
results.to_csv("chain_tailored_thoughts_results.csv", mode='a', header=not pd.io.common.file_exists("chain_tailored_thoughts_results.csv"), index=False)
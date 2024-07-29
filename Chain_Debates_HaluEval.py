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
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# TODO: Set parameters for the number of rows to analyse
n = 25

# Ignore the rows already analyzed (experiment 1 did 6000)
df = df.iloc[6000:]

# Initialize lists to store true labels and predicted labels for F1 score calculation
true_labels = []
baseline_preds = []
phi3_chain_debates_preds = []
gpt4o_mini_chain_debates_preds = []
gpt4o_chain_debates_preds = []

for i in range(n):
    row = df.iloc[i]
    document = row['document']
    right_summary = row['right_summary']
    hallucinated_summary = row['hallucinated_summary']
    
    results_per_iteration = pd.DataFrame(columns=[
    'Row', 'Document', 'Right Summary', 'Hallucinated Summary', 
    'True Label', 'Baseline Prediction', 'Phi3 Chain Debates Prediction', 
    'GPT4o Mini Chain Debates Prediction', 'GPT4o Chain Debates Prediction'
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
    
    baseline_pred = baseline(document, right_summary)
    baseline_preds.append(baseline_pred)
    
    #(phi3_chain_debates_pred, phi3_debate_hallucinated, phi3_debate_supported) = chain_debates("phi3", document, right_summary)
    #phi3_chain_debates_preds.append(phi3_chain_debates_pred)
        
    (gpt4o_mini_chain_debates_pred, gpt4o_mini_debate_hallucinated, gpt4o_mini_debate_supported) = chain_debates("gpt4o_mini", document, right_summary)
    gpt4o_mini_chain_debates_preds.append(gpt4o_mini_chain_debates_pred)
    
    (gpt4o_chain_debates_pred, gpt4o_debate_hallucinated, gpt4o_debate_supported) = chain_debates("gpt4o", document, right_summary)
    gpt4o_chain_debates_preds.append(gpt4o_chain_debates_pred)
    
    # Save the results of the current iteration for the true summary
    results_per_iteration = pd.concat([results_per_iteration, pd.DataFrame([{
        'Row': i+1, 
        'Document': document, 
        'Right Summary': right_summary, 
        'Hallucinated Summary': '',
        'True Label': 0,
        'Baseline Prediction': baseline_pred,
        #'Phi3 Chain Debates Prediction': phi3_chain_debates_pred,
        #'Phi3 Debate Hallucinated': phi3_debate_hallucinated,
        #'Phi3 Debate History': phi3_debate_supported,
        'GPT4o Mini Chain Debates Prediction': gpt4o_mini_chain_debates_pred,
        'GPT4o Mini Debate Hallucinated': gpt4o_mini_debate_hallucinated,
        'GPT4o Mini Debate History': gpt4o_mini_debate_supported,
        'GPT4o Chain Debates Prediction': gpt4o_chain_debates_pred,
        'GPT4o Debate Hallucinated': gpt4o_debate_hallucinated,
        'GPT4o Debate History': gpt4o_debate_supported
    }])], ignore_index=True)
    
    # Run the analysis for hallucinated summary
    print("-" * 100)
    print("Running the process for a hallucinated summary")
    true_labels.append(1) 
    
    baseline_pred = baseline(document, hallucinated_summary)
    baseline_preds.append(baseline_pred)
    
    #(phi3_chain_debates_pred, phi3_debate_hallucinated, phi3_debate_supported) = chain_debates("phi3", document, hallucinated_summary)
    #phi3_chain_debates_preds.append(phi3_chain_debates_pred)
        
    (gpt4o_mini_chain_debates_pred, gpt4o_mini_debate_hallucinated, gpt4o_mini_debate_supported) = chain_debates("gpt4o_mini", document, hallucinated_summary)
    gpt4o_mini_chain_debates_preds.append(gpt4o_mini_chain_debates_pred)
    
    (gpt4o_chain_debates_pred, gpt4o_debate_hallucinated, gpt4o_debate_supported) = chain_debates("gpt4o", document, hallucinated_summary)
    gpt4o_chain_debates_preds.append(gpt4o_chain_debates_pred)
    
    # Save the results of the current iteration for the hallucinated summary
    results_per_iteration = pd.concat([results_per_iteration, pd.DataFrame([{
        'Row': i+1, 
        'Document': document, 
        'Right Summary': '',
        'Hallucinated Summary': hallucinated_summary,
        'True Label': 1,
        'Baseline Prediction': baseline_pred,
        #'Phi3 Chain Debates Prediction': phi3_chain_debates_pred,
        #'Phi3 Debate Hallucinated': phi3_debate_hallucinated,
        #'Phi3 Debate History': phi3_debate_supported,
        'GPT4o Mini Chain Debates Prediction': gpt4o_mini_chain_debates_pred,
        'GPT4o Mini Debate Hallucinated': gpt4o_mini_debate_hallucinated,
        'GPT4o Mini Debate History': gpt4o_mini_debate_supported,
        'GPT4o Chain Debates Prediction': gpt4o_chain_debates_pred,
        'GPT4o Debate Hallucinated': gpt4o_debate_hallucinated,
        'GPT4o Debate History': gpt4o_debate_supported
    }])], ignore_index=True)
    
    # Save the current iteration results to a CSV file
    #results_per_iteration.to_csv("chain_debates.csv", mode='a', header=not pd.io.common.file_exists("chain_debates.csv"), index=False)


print("-" * 100)
print(f"Results after {n} iterations:\n")

# Calculate and print the accuracies
accuracies = {}

accuracies['baseline_TNR'] = sum([1 for i in range(0, 2*n, 2) if baseline_preds[i] == 0]) / n
accuracies['baseline_TPR'] = sum([1 for i in range(1, 2*n, 2) if baseline_preds[i] == 1]) / n

accuracies['phi3_chain_debates_TNR'] = sum([1 for i in range(0, 2*n, 2) if phi3_chain_debates_preds[i] == 0]) / n
accuracies['phi3_chain_debates_TPR'] = sum([1 for i in range(1, 2*n, 2) if phi3_chain_debates_preds[i] == 1]) / n

accuracies['gpt4o_mini_chain_debates_TNR'] = sum([1 for i in range(0, 2*n, 2) if gpt4o_mini_chain_debates_preds[i] == 0]) / n
accuracies['gpt4o_mini_chain_debates_TPR'] = sum([1 for i in range(1, 2*n, 2) if gpt4o_mini_chain_debates_preds[i] == 1]) / n

accuracies['gpt4o_chain_debates_TNR'] = sum([1 for i in range(0, 2*n, 2) if gpt4o_chain_debates_preds[i] == 0]) / n
accuracies['gpt4o_chain_debates_TPR'] = sum([1 for i in range(1, 2*n, 2) if gpt4o_chain_debates_preds[i] == 1]) / n

print("Baseline (TNR):", accuracies['baseline_TNR'])
print("Baseline (TPR):", accuracies['baseline_TPR'])
print("Phi3 Chain Debates (TNR):", accuracies['phi3_chain_debates_TNR'])
print("Phi3 Chain Debates (TPR):", accuracies['phi3_chain_debates_TPR'])
print("GPT4o Mini Chain Debates (TNR):", accuracies['gpt4o_mini_chain_debates_TNR'])
print("GPT4o Mini Chain Debates (TPR):", accuracies['gpt4o_mini_chain_debates_TPR'])
print("GPT4o Chain Debates (TNR):", accuracies['gpt4o_chain_debates_TNR'])
print("GPT4o Chain Debates (TPR):", accuracies['gpt4o_chain_debates_TPR'])

# Calculate and print the F1 scores
baseline_f1 = f1_score(true_labels, baseline_preds)
phi3_chain_debates_f1 = f1_score(true_labels, phi3_chain_debates_preds)
gpt4o_mini_chain_debates_f1 = f1_score(true_labels, gpt4o_mini_chain_debates_preds)
gpt4o_chain_debates_f1 = f1_score(true_labels, gpt4o_chain_debates_preds)

print("Baseline F1 Score:", baseline_f1)
print("Phi3 Chain Debates F1 Score:", phi3_chain_debates_f1)
print("GPT4o Mini Chain Debates F1 Score:", gpt4o_mini_chain_debates_f1)
print("GPT4o Chain Debates F1 Score:", gpt4o_chain_debates_f1)

# Create a DataFrame for the new results
results = pd.DataFrame({
    'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    'Number of Rows': [n],
    'Baseline F1 Score': [baseline_f1],
    'Phi3 Chain Debates F1 Score': [phi3_chain_debates_f1],
    'GPT4o Mini Chain Debates F1 Score': [gpt4o_mini_chain_debates_f1],
    'GPT4o Chain Debates F1 Score': [gpt4o_chain_debates_f1],
    'Baseline (TPR)': [accuracies['baseline_TPR']],
    'Baseline (TNR)': [accuracies['baseline_TNR']],
    'Phi3 Chain Debates (TPR)': [accuracies['phi3_chain_debates_TPR']],
    'Phi3 Chain Debates (TNR)': [accuracies['phi3_chain_debates_TNR']],
    'GPT4o Mini Chain Debates (TPR)': [accuracies['gpt4o_mini_chain_debates_TPR']],
    'GPT4o Mini Chain Debates (TNR)': [accuracies['gpt4o_mini_chain_debates_TNR']],
    'GPT4o Chain Debates (TPR)': [accuracies['gpt4o_chain_debates_TPR']],
    'GPT4o Chain Debates (TNR)': [accuracies['gpt4o_chain_debates_TNR']]
})

# Append the new results to the existing CSV file
#results.to_csv("chain_debates_results.csv", mode='a', header=not pd.io.common.file_exists("chain_debates_results.csv"), index=False)
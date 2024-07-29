from helpers import *
from datasets import load_dataset
from sklearn.metrics import f1_score
import pandas as pd
from datetime import datetime


# Initialize lists to store true labels and predicted labels for F1 score calculation
true_labels = []
baseline_preds = []
phi3_sentence_level_preds = []
gpt4o_mini_sentence_level_preds = []
gpt4o_sentence_level_preds = []

# Read the csv
results_per_iteration = pd.read_csv("counterfactual_debate.csv")

for i in range(50):    
    # Take row i
    row = results_per_iteration.iloc[i]    
    
    # Extract pregenerated debates
    debate_phi3 = "\n" + row['Phi3 Debate Hallucinated'] + "\n" + row['Phi3 Debate Supported'] + "\n"
    debate_gpt4o_mini = "\n" + row['GPT4o Mini Debate Hallucinated'] + "\n" + row['GPT4o Mini Debate Supported'] + "\n"
    debate_gpt4o = "\n" + row['GPT4o Debate Hallucinated'] + "\n" + row['GPT4o Debate Supported'] + "\n"
    
    print("-" * 100)
    print("ANALYZING ROW", i+1, "OF", 50, "...")
    print("-" * 100)

    # Print the values of the selected row
    document = row['Document']
    print("Document:", document)
    
    right_summary = row['Right Summary']
    halucinated_summary = row['Hallucinated Summary']

    if i % 2 == 0:
        print("Right Summary:", right_summary)
        print("-" * 100)
        
        # Run the analysis for right summary
        print("Running the process for a supported summary")
        true_labels.append(0)
        
        baseline_pred = baseline(document, right_summary)
        baseline_preds.append(baseline_pred)
        
        phi3_sentence_level_pred = counterfactual_debate_extended(document, right_summary, debate_phi3)
        phi3_sentence_level_preds.append(phi3_sentence_level_pred)
            
        gpt4o_mini_sentence_level_pred = counterfactual_debate_extended(document, right_summary, debate_gpt4o_mini)
        gpt4o_mini_sentence_level_preds.append(gpt4o_mini_sentence_level_pred)
        
        gpt4o_sentence_level_pred = counterfactual_debate_extended(document, right_summary, debate_gpt4o)
        gpt4o_sentence_level_preds.append(gpt4o_sentence_level_pred)
        
    else:
        print('Hallucinated Summary:', halucinated_summary)
        print("-" * 100)
        
        # Run the analysis for hallucinated summary
        print("Running the process for a hallucinated summary")
        true_labels.append(1)
        
        baseline_pred = baseline(document, halucinated_summary)
        baseline_preds.append(baseline_pred)
        
        phi3_sentence_level_pred = counterfactual_debate_extended(document, halucinated_summary, debate_phi3)
        phi3_sentence_level_preds.append(phi3_sentence_level_pred)
        
        gpt4o_mini_sentence_level_pred = counterfactual_debate_extended(document, halucinated_summary, debate_gpt4o_mini)
        gpt4o_mini_sentence_level_preds.append(gpt4o_mini_sentence_level_pred)
        
        gpt4o_sentence_level_pred = counterfactual_debate_extended(document, halucinated_summary, debate_gpt4o)
        gpt4o_sentence_level_preds.append(gpt4o_sentence_level_pred)

print("-" * 100)
print(f"Results after 50 iterations:\n")

# Calculate and print the accuracies
accuracies = {}

accuracies['baseline_TNR'] = sum([1 for i in range(0,  50, 2) if baseline_preds[i] == 0]) / 25
accuracies['baseline_TPR'] = sum([1 for i in range(1,  50, 2) if baseline_preds[i] == 1]) / 25

accuracies['phi3_sentence_level_TNR'] = sum([1 for i in range(0,  50, 2) if phi3_sentence_level_preds[i] == 0]) / 25
accuracies['phi3_sentence_level_TPR'] = sum([1 for i in range(1,  50, 2) if phi3_sentence_level_preds[i] == 1]) / 25

accuracies['gpt4o_mini_sentence_level_TNR'] = sum([1 for i in range(0,  50, 2) if gpt4o_mini_sentence_level_preds[i] == 0]) / 25
accuracies['gpt4o_mini_sentence_level_TPR'] = sum([1 for i in range(1,  50, 2) if gpt4o_mini_sentence_level_preds[i] == 1]) / 25

accuracies['gpt4o_sentence_level_TNR'] = sum([1 for i in range(0,  50, 2) if gpt4o_sentence_level_preds[i] == 0]) / 25
accuracies['gpt4o_sentence_level_TPR'] = sum([1 for i in range(1,  50, 2) if gpt4o_sentence_level_preds[i] == 1]) / 25

print("Baseline (TNR):", accuracies['baseline_TNR'])
print("Baseline (TPR):", accuracies['baseline_TPR'])
print("Phi3 Sentence Level (TNR):", accuracies['phi3_sentence_level_TNR'])
print("Phi3 Sentence Level (TPR):", accuracies['phi3_sentence_level_TPR'])
print("GPT4o Mini Sentence Level (TNR):", accuracies['gpt4o_mini_sentence_level_TNR'])
print("GPT4o Mini Sentence Level (TPR):", accuracies['gpt4o_mini_sentence_level_TPR'])
print("GPT4o Sentence Level (TNR):", accuracies['gpt4o_sentence_level_TNR'])
print("GPT4o Sentence Level (TPR):", accuracies['gpt4o_sentence_level_TPR'])

# Calculate and print the F1 scores
baseline_f1 = f1_score(true_labels, baseline_preds)
phi3_sentence_level_f1 = f1_score(true_labels, phi3_sentence_level_preds)
gpt4o_mini_sentence_level_f1 = f1_score(true_labels, gpt4o_mini_sentence_level_preds)
gpt4o_sentence_level_f1 = f1_score(true_labels, gpt4o_sentence_level_preds)

print("Baseline F1 Score:", baseline_f1)
print("Phi3 Sentence Level F1 Score:", phi3_sentence_level_f1)
print("GPT4o Mini Sentence Level F1 Score:", gpt4o_mini_sentence_level_f1)
print("GPT4o Sentence Level F1 Score:", gpt4o_sentence_level_f1)

# Create a DataFrame for the new results
results = pd.DataFrame({
    'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    'Baseline F1 Score': [baseline_f1],
    'Phi3 Sentence Level F1 Score': [phi3_sentence_level_f1],
    'GPT4o Mini Sentence Level F1 Score': [gpt4o_mini_sentence_level_f1],
    'GPT4o Sentence Level F1 Score': [gpt4o_sentence_level_f1],
    'Baseline (TPR)': [accuracies['baseline_TPR']],
    'Baseline (TNR)': [accuracies['baseline_TNR']],
    'Phi3 Sentence Level (TPR)': [accuracies['phi3_sentence_level_TPR']],
    'Phi3 Sentence Level (TNR)': [accuracies['phi3_sentence_level_TNR']],
    'GPT4o Mini Sentence Level (TPR)': [accuracies['gpt4o_mini_sentence_level_TPR']],
    'GPT4o Mini Sentence Level (TNR)': [accuracies['gpt4o_mini_sentence_level_TNR']],
    'GPT4o Sentence Level (TPR)': [accuracies['gpt4o_sentence_level_TPR']],
    'GPT4o Sentence Level (TNR)': [accuracies['gpt4o_sentence_level_TNR']]
})

# Append the new results to the existing CSV file
results.to_csv("counterfactual_debate_extended_results.csv", mode='a', header=not pd.io.common.file_exists("counterfactual_debate_extended_results.csv"), index=False)

        
import json
import pandas as pd

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

json_data = read_jsonl("Code/Data/QAGS/mturk_cnndm.jsonl")

# Create a list to hold the data
data = []

# Iterate over each item in the JSON data
for item in json_data:
    article = item["article"]
    for sentence_data in item["summary_sentences"]:
        summary_sentence = sentence_data["sentence"]
        responses = [response["response"] for response in sentence_data["responses"]]
        if len(responses) >= 3:
            responses = responses[:3]
        else:
            responses += [None] * (3 - len(responses))
        row = {
            "article": article,
            "summary": ' '.join([sd["sentence"] for sd in item["summary_sentences"]]),
            "highlighted_sentence": summary_sentence,
            "worker_1": responses[0],
            "worker_2": responses[1],
            "worker_3": responses[2],
        }
        data.append(row)

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(data)

# Step 1: Filter rows with consistent responses
def all_same(lst):
    # Count the occurrences of 'yes' and 'no'
    count_yes = 0
    count_no = 0
    
    for response in lst:
        if response == 'yes':
            count_yes += 1
        elif response == 'no':
            count_no += 1
    
    # Return True if all are 'yes' or at least two are 'no'
    return count_yes == 3 or count_no >= 2

df['consistent'] = df[['worker_1', 'worker_2', 'worker_3']].apply(lambda row: all_same(row), axis=1)

def articles_with_consistent_responses(df):
    return df.groupby('article').filter(lambda x: x['consistent'].all())

df_filtered = articles_with_consistent_responses(df)
print(df_filtered)

# Step 2: Classify each article
def determine_class(group):
    responses = group[['worker_1', 'worker_2', 'worker_3']].values
    if all(all(r == 'yes' for r in response) for response in responses):
        return 'correct'
    else:
        return 'hallucinated'

def classify_article(group):
    classification = determine_class(group)
    summary = group['summary'].iloc[0]
    return pd.Series({'classification': classification, 'summary': summary})

df_classification = df_filtered.groupby('article').apply(classify_article).reset_index()
print(df_classification)
print(df_classification['classification'].value_counts())

# Step 3: Save results to CSV files
df_correct = df_classification[df_classification['classification'] == 'correct']
df_hallucinated = df_classification[df_classification['classification'] == 'hallucinated']

df_correct.to_csv('correct.csv', index=False, header=True)
df_hallucinated.to_csv('hallucinated.csv', index=False, header=True)

print("Data saved to 'correct.csv' and 'hallucinated.csv'")
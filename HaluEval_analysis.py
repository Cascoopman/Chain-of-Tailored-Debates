import pandas as pd

df1 = pd.read_csv('Code/Data/QAGS/correct.csv')
df2 = pd.read_csv('Code/Data/QAGS/hallucinated.csv')

# add the df's together
df = pd.concat([df1, df2], ignore_index=True)
print(len(df))
print(df.columns)

# calculate length of articles and summaries
length_articles = 0
length_summaries = 0

for i in range(len(df)):
    length_articles += len(df['article'][i])
    length_summaries += len(df['summary'][i])
    
print('Average length of articles:', length_articles/len(df))
print('Average length of summaries:', length_summaries/len(df))
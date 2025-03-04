import pandas as pd

# load the csv data set for business and health data https://archive.ics.uci.edu/dataset/359/news+aggregator
df_csv = pd.read_csv('./dataset/data.csv', sep='\t', header=None)

# assign column names
df_csv.columns = ['ID', 'Title', 'URL', 'Publisher', 'Category', 'Story', 'Hostname','Timestamp']

# filter rows where category is 'b' (business) or 'm' (health)
df_filtered = df_csv[df_csv['Category'].isin(['b', 'm'])]

# select only the required columns
df_filtered = df_filtered[['Title', 'URL', 'Category']]

# load the json data for politics data
# https://www.kaggle.com/datasets/rmisra/news-category-dataset
# https://rishabhmisra.github.io/publications/ , News Category Dataset
df_json = pd.read_json('./dataset/data.json', lines=True)

# filter for category 'POLITICS' and set as 'p'
df_json_filtered = df_json[df_json['category'] == 'POLITICS'].copy()
df_json_filtered['category'] = 'p'

# rename the columns to match the required format
df_json_filtered = df_json_filtered.rename(columns={'headline': 'Title', 'link':'URL', 'category': 'Category'})

# select the required columns
df_json_filtered = df_json_filtered[['Title', 'URL', 'Category']]

# combine both datasets
df_filtered = pd.concat([df_filtered, df_json_filtered], ignore_index=True)

# save the combined dataset to a new file
df_filtered.to_csv('./dataset/filtered_data.csv', index=False, sep='\t')


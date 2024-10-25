import torch
import numpy as np 
import pandas as pd
import re 
import os 

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, speech_ids, speeches):
        self.speech_ids = speech_ids
        self.speeches = speeches

    def __len__(self):
        return len(self.speeches)

    def __getitem__(self, index):
        speech_id = self.speech_ids[index]
        speeches = self.speeches[index]
        return speech_id, speeches

    def get_dataloader(self, batch_size=2048, shuffle=False):
      return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

def get_prominent_senses(word, window_size=20, threshold=0.2):
    word_tibble = pd.read_csv(f"./clusters/{word}_cluster_counts.csv")
    cluster_columns = [col for col in word_tibble.columns if col.startswith("cluster_") and not col.endswith("_p")]
    
    def max_proportion_in_window(column):
        max_proportion = 0
        years = word_tibble['year'].unique()
        for start_year in years:
            end_year = start_year + window_size
            window_data = word_tibble[(word_tibble['year'] >= start_year) & (word_tibble['year'] < end_year)]
            proportion = window_data[column].sum() / window_data['n'].sum()
            if proportion > max_proportion:
                max_proportion = proportion
        return max_proportion
    
    # Determine which columns to include
    columns_to_include = [col for col in cluster_columns if max_proportion_in_window(col) >= threshold]
    
    # Strip the column names of everything but the number
    cluster_numbers = [int(re.search(r'\d+', col).group()) for col in columns_to_include]
    
    return cluster_numbers

# Load the data (assuming you have already loaded the DataFrame `word_tibble`)
# word_tibble = pd.read_csv("clustering_results/{word}/{word}_cluster_counts.csv")
# prominent_clusters = get_prominent_senses(word_tibble)
# print(prominent_clusters)


def get_high_change_senses(word, window_size=10, threshold=0.3):
    word_df = pd.read_csv(f"./clusters/{word}_cluster_counts.csv")
    cluster_p_columns = [col for col in word_df.columns if col.startswith("cluster_") and col.endswith("_p")]
    high_change_columns = []
    for column in cluster_p_columns:
        years = word_df['year'].unique()
        mean_props = []
        for start_year in years:
            end_year = start_year + window_size
            if end_year <= max(years):
                window_data = word_df[(word_df['year'] >= start_year) & (word_df['year'] < end_year)]
                mean_prop = np.mean(window_data[column])
                mean_props.append(mean_prop)
            else:
                break
        max_prop = max(mean_props)
        min_prop = min(mean_props)
        if max_prop - min_prop > threshold:
            high_change_columns.append(column)
    # 
    cluster_numbers = [re.search(r'\d+', col).group() for col in high_change_columns]
    return cluster_numbers



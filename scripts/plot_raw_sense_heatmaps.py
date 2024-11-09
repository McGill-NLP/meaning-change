import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.data import get_high_change_senses
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.ndimage import gaussian_filter


plot_directory = './plots/sense_prob_heatmaps'
if not os.path.isdir(plot_directory):
    os.mkdir(plot_directory)

clustering_results_dir = './clusters/'

# Initializing word list:
word_list_path = './assets/wordlist.txt'
word_list = []
with open(word_list_path, 'r') as file:
    for line in file:
        word_list.append(line.strip())

# Get the senses we actually modelled, i.e. those with high change:
print("Collecting word senses that showed high change...")
word_senses = {}
for word in tqdm(word_list):
    high_change_senses = get_high_change_senses(word, window_size=20, threshold=0.30)
    if len(high_change_senses) > 0:
        word_senses[word] = high_change_senses

print("Done!")

print("Making plots...")
for word in tqdm(word_senses.keys()): # So as to only iterate over words that actually had senses showing significant change
    word_cluster_data_path = os.path.join(clustering_results_dir, f"{word}_cluster_counts.csv")
    word_cluster_data = pd.read_csv(word_cluster_data_path)
    relevant_clusters = word_senses[word]
    for cluster in relevant_clusters:
        relevant_cluster_column = f"cluster_{cluster}_p" # We can consider raw substitution counts later perhaps
        word_cluster_data_filtered = word_cluster_data[[column for column in word_cluster_data.columns if not 'cluster' in column or column==relevant_cluster_column]]
        #
        filename = os.path.join(plot_directory, f"{word}-{cluster}.png")
        #
        pivoted_data = word_cluster_data_filtered.pivot_table(index='year', columns='age', values=relevant_cluster_column, aggfunc='mean')
        smooth_values = gaussian_filter(pivoted_data.values, sigma=0.1)
        smoothed_data = pd.DataFrame(smooth_values, index=pivoted_data.index, columns=pivoted_data.columns)
        #
        vmin = 0 
        vmax = 1 
        cmap = 'cool' # Change to see what different colormaps look like
        mask = None # Change to be NaNs in smoothed data to control background colour
        #
        plt.figure(figsize=(10, 8))  # Adjust the size as needed
        sns.heatmap(smoothed_data, cmap=cmap, cbar_kws={'label': f'Probability of Sense {cluster}'}, vmin=vmin, vmax=vmax, mask=mask)
        # plt.gca().set_facecolor('#000000') # Sets colour of background tiles -- useful if using custom 
        plt.title(f"Likelihood of Sense {cluster} of '{word}'")
        plt.xlabel('Age')
        plt.ylabel('Speech Year')
        plt.gca().invert_yaxis()
        # Save the plot
        plt.savefig(filename)
        plt.close()

print("All done!")
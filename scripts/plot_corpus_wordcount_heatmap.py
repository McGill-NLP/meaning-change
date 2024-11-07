import os
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.ndimage import gaussian_filter

# These are used to filter the corpus -- same default parameter values as used in the rest of the codebase.
min_age = 25
procedural_prob_threshold = 0.5

plot_directory = './plots/misc'
if not os.path.isdir(plot_directory):
    os.mkdir(plot_directory)

filename = os.path.join(plot_directory, 'corpus_wordcount_heatmap.png')

corpus_path = "./datasets/uscongress_base_preprocessed.csv"
print("Reading in corpus...")
corpus = pd.read_csv(corpus_path)
print("Done!")

print("Filtering corpus...")
# Remove speeches deemed to have too high a probability of being procedural (0.5 here):
corpus = corpus[corpus['procedural_prob']<procedural_prob_threshold]

print("Done!")

print("Getting speech word counts...")
corpus['speech_length'] = corpus['speech'].apply(lambda x: len(x.strip().split()))
print("Done!")

pivoted_data = corpus.pivot_table(index='year', columns='age', values='speech_length', aggfunc='sum')
sns.set(font_scale=1.5)
plt.figure(figsize=(12, 9))  # Adjust the size as needed
sns.heatmap(pivoted_data, cmap='cool')
plt.title(f"Total Words in Speeches")
plt.xlabel('Age')
plt.ylabel('Speech Year')
plt.gca().invert_yaxis()
# Save the plot
plt.savefig(filename)
plt.close()

print("All done!")
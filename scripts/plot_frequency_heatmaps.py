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


word_list_path = './assets/wordlist.txt'
word_list = []
with open(word_list_path, 'r') as file:
    for line in file:
        word_list.append(line.strip())

plot_directory = './plots/frequency_heatmaps'
if not os.path.isdir(plot_directory):
    os.mkdir(plot_directory)


def get_word_counts_year_yob_wise(speeches, tqdm=True):
    counts = Counter()
    if tqdm==True:
        for speech in tqdm(speeches):
            words = speech.split()
            counts.update(words)
    else:
        for speech in speeches:
            words = speech.split()
            counts.update(words)
    df = pd.DataFrame(counts.values(), columns=['raw_count'], index=counts.keys()).sort_values(by='raw_count', ascending=False)
    total_words = sum(df['raw_count'])
    df['log_freq'] = np.log(df['raw_count']) - np.log(total_words)
    return df  

def year_wise_to_word_wise(words, year_wise_dict):
    dictlist_for_df = []
    years = np.unique([key[4:8] for key in year_wise_dict.keys()])
    yobs = np.unique([key[-4:] for key in year_wise_dict.keys()])
    for word in tqdm(words):
        for year in years:
            for yob in yobs:
                key = f'year{year}_yob{yob}'
                try:
                    data = year_wise_dict[key].loc[word]
                    raw_count = data['raw_count']
                    log_freq = data['log_freq']
                    word_dict = {'word':word, 'year':int(year), 'yob':int(yob), 'age':int(year)-int(yob), 'raw_count': raw_count, 'log_freq': log_freq}
                    dictlist_for_df.append(word_dict)
                except:
                    continue 
    return pd.DataFrame(dictlist_for_df)


corpus_path = "./datasets/uscongress_base_preprocessed.csv"
print("Reading in corpus...")
corpus = pd.read_csv(corpus_path)
print("Done!")

print("Filtering corpus...")
# Remove any cases of age being less than 25, since these are presumably errors:
corpus = corpus[corpus['age']>=min_age]
# Remove speeches deemed to have too high a probability of being procedural (0.5 here):
corpus = corpus[corpus['procedural_prob']<procedural_prob_threshold]
print("Done!")

print("Fetching base counts across the corpus...")
years = np.unique(corpus['year'])
year_yob_wise_word_counts = {}
for year in tqdm(years):
    # print(f"Starting on speeches from {year}")
    year_indexed = corpus[corpus['year']==year]
    for yob in np.unique(year_indexed['yob']):
        key = f'year{str(year)}_yob{str(yob)}'
        year_yob_indexed = year_indexed[year_indexed['yob']==yob]
        year_yob_wise_word_counts[key] = get_word_counts_year_yob_wise(year_yob_indexed['speech'], tqdm=False)

print("Done!")

print("Processing to get word-wise usage frequency data...")
word_wise_counts = year_wise_to_word_wise(word_list, year_yob_wise_word_counts)
print("Done!")

print("Starting on processing plots...")
max_vals = []
filler_vals = []
for word in word_list:
    word_specific = word_wise_counts[word_wise_counts['word'] == word]
    max_val = np.max(word_specific['log_freq'])
    max_vals.append(max_val)
    filler_val = np.min(word_specific['log_freq']) - np.std(word_specific['log_freq'])
    filler_vals.append(filler_val)

upper_lim_global = np.max(max_vals) # Upper limit for global colour scale below
filler_val_global = np.min(filler_vals) # Filler ('background') colour for heatmap -- also serves as lower limit for global colour scale
for word in tqdm(word_list):
    try:
        filename = os.path.join(plot_directory, f"{word}.png")
        filename_rel = os.path.join(plot_directory, f"{word}_rel.png")
        word_specific = word_wise_counts[word_wise_counts['word'] == word]
        filler_val_word_specific = np.min(word_specific['log_freq']) - np.std(word_specific['log_freq'])  # Fixed to 'word_specific'
        pivoted_data = word_specific.pivot_table(index='year', columns='age', values='log_freq', aggfunc='mean')  # Using pivot_table to handle duplicates
        
        # Use this to play with heatmap color scheme:
        turquoise_yellow = LinearSegmentedColormap.from_list("turquoise_yellow", ["turquoise", "yellow"])
        
        
        # Making a heatmap with an invariant colour scale:
        pivoted_data_filled = pivoted_data.fillna(filler_val_global)
        smooth_values = gaussian_filter(pivoted_data_filled.values, sigma=0.5)
        smoothed_data = pd.DataFrame(smooth_values, index=pivoted_data_filled.index, columns=pivoted_data_filled.columns)
        vmin = filler_val_global  # Replace with your desired minimum value
        vmax = upper_lim_global  # Replace with your desired maximum value
        #
        plt.figure(figsize=(10, 8))  # Adjust the size as needed
        sns.heatmap(smoothed_data, cmap=turquoise_yellow, cbar_kws={'label': 'Log proportion of total words spoken'}, vmin=vmin, vmax=vmax)
        plt.title(f"Usage frequency of '{word}'")
        plt.xlabel('Age')
        plt.ylabel('Speech Year')
        plt.gca().invert_yaxis()
        # Save the plot
        plt.savefig(filename)
        plt.close()
        
        
        # Making a heatmap with a variant colour scale:
        pivoted_data_filled = pivoted_data.fillna(filler_val_word_specific)
        smooth_values = gaussian_filter(pivoted_data_filled.values, sigma=0.5)
        smoothed_data = pd.DataFrame(smooth_values, index=pivoted_data_filled.index, columns=pivoted_data_filled.columns)
        vmin = filler_val_word_specific  # Replace with your desired minimum value
        vmax = None  # Replace with your desired maximum value
        #
        plt.figure(figsize=(10, 8))  # Adjust the size as needed
        sns.heatmap(smoothed_data, cmap=turquoise_yellow, cbar_kws={'label': 'Log proportion of total words spoken'}, vmin=vmin, vmax=vmax)
        plt.title(f"Usage frequency of '{word}'")
        plt.xlabel('Age')
        plt.ylabel('Speech Year')
        plt.gca().invert_yaxis()
        # Save the plot
        plt.savefig(filename_rel)
        plt.close()
    except:
        continue 

print("All done!")
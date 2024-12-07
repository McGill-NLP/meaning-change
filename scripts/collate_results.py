import os 
import re
# from scripts.utils.data import get_high_change_senses
from tqdm import tqdm 
import pandas as pd 
import numpy as np
import json

# Initializing word list:
word_list_path = './assets/wordlist.txt'
word_list = []
with open(word_list_path, 'r') as file:
    for line in file:
        word_list.append(line.strip())

# Getting a-optimization results:
a_optimization_results_directory = './a-optimization_results'
a_optimization_results = {}
for word in word_list:
    json_file = f"a-optim-{word}.json"
    full_filepath = os.path.join(a_optimization_results_directory, json_file)
    if os.path.exists(full_filepath):
        with open(full_filepath, 'r') as json_file:
            a_optimization_results[word] = json.load(json_file)

# Reading in and processing a-optimization results: a coefs, C.I.s, whether a-coef is significantly non-zero: 
processed_data = [datapoint for sublist in a_optimization_results.values() for datapoint in sublist] # Remove first nested list structure so can be read in as df
processed_data = pd.DataFrame(processed_data) # Read in as df
for column in processed_data.columns:
    processed_data[column] = processed_data[column].apply(lambda x: x[0]) # Remove lower nested list structure so data directly accessible

for column in ['ci_lower', 'ci_upper']:
    processed_data[column] = processed_data[column].apply(lambda x: None if x=='NaN' else x)

processed_data['ci_width'] = processed_data['ci_upper'] - processed_data['ci_lower']
processed_data['sig_non_zero'] = ~((processed_data['ci_lower'] < 0) & (processed_data['ci_upper'] > 0))


# Reading in GAMM results, adding them to the df:
def extract_key_metrics(model_summary_path):
    r_sq_regex = re.compile(r'R-sq\.\(adj\)\s=\s+([\d\.]+)')
    freml_regex = re.compile(r'fREML\s=\s+([\d\.]+)')
    dev_exp_regex = re.compile(r'Deviance\sexplained\s=\s+([\d\.]+)')
    with open(model_summary_path, 'r') as file:
        summary_string = file.read()
        r_sq = float(r_sq_regex.findall(summary_string)[0])
        freml = float(freml_regex.findall(summary_string)[0])
        dev_exp = float(dev_exp_regex.findall(summary_string)[0])/100
    return r_sq, freml, dev_exp

GAMM_plots_and_summaries_dir_LO = './plots/LO_GAMM_plots_and_summaries'
GAMM_plots_and_summaries_dir_2D = './plots/2D_GAMM_plots_and_summaries'

def get_model_stats(row):
    # Initialize as NA values:
    r_sq_LO, freml_LO, dev_exp_LO, r_sq_2D, freml_2D, dev_exp_2D = None, None, None, None, None, None
    
    word = row['word']
    sense = row['sense']
    GAMM_summary_path_LO = os.path.join(GAMM_plots_and_summaries_dir_LO, f"{word}-{sense}-summary.txt")
    GAMM_summary_path_2D = os.path.join(GAMM_plots_and_summaries_dir_2D, f"{word}-{sense}-summary.txt")
    
    # Actual value only if files exist:
    if os.path.exists(GAMM_summary_path_LO):
        r_sq_LO, freml_LO, dev_exp_LO = extract_key_metrics(GAMM_summary_path_LO)
    if os.path.exists(GAMM_summary_path_2D):
        r_sq_2D, freml_2D, dev_exp_2D = extract_key_metrics(GAMM_summary_path_2D)
    
    # ^ This should keep things flexible with word-sense combinations that haven't finished yet
    return r_sq_LO, freml_LO, dev_exp_LO, r_sq_2D, freml_2D, dev_exp_2D

GAMM_comparison_datapoints_raw = processed_data.apply(lambda row: get_model_stats(row), axis=1)
GAMM_comparison_columns = ['r_sq_LO', 'freml_LO', 'dev_exp_LO', 'r_sq_2D', 'freml_2D', 'dev_exp_2D']
for i in range(len(GAMM_comparison_columns)):
    processed_data[GAMM_comparison_columns[i]] = [x[i] for x in GAMM_comparison_datapoints_raw]

# Adding a column for overall word freq (Qs: do we get more tight C.I.s with more data? Are negative a coefs from sparse data?)
substitutions_dir = './substitutions/' # Instead of reading in the whole speech dataset (heavy), we can compute original word counts from the substitution data
def get_overall_word_freq(word):
    substitutions_results = []
    with open(os.path.join(substitutions_dir, f"{word}.jsonl"), 'r') as json_file:
        for line in json_file:
            substitutions_results.append(json.loads(line))
    #
    word_counts = [len(speech_datapoint['substitutions']) for speech_datapoint in substitutions_results]
    # ^^^ Note on this line of code above:
    # Each entry in the results data corresponds to one speech. 
    # For each speech, 'substitutions' is a list (of lists of substitutions) whose length corresponds to the number of word mentions in the speech.
    # Each sublist in turn corresponds to the model-generated substitutes for that instance of the word in the speech.
    return np.sum(word_counts)

processed_data['overall_word_freq'] = processed_data['word'].apply(lambda x: get_overall_word_freq(x))

# Add a column for sense-specific substitution counts (Qs: similar to above, but sense-specific)
clustering_results_dir = './clusters/'
def get_cluster_substitution_count(word, sense):
    word_cluster_data = pd.read_csv(os.path.join(clustering_results_dir, f"{word}_cluster_counts.csv"))
    cluster_substitution_count = np.sum(word_cluster_data[f"cluster_{sense}"])
    return cluster_substitution_count

def get_cluster_substitution_count_from_row(row):
    word = row['word']
    sense = row['sense']
    return get_cluster_substitution_count(word, sense)

processed_data['cluster_substitution_total'] = processed_data.apply(lambda x: get_cluster_substitution_count_from_row(x), axis=1)

# Add a column for normalized L2 Distances between 2D and LO GAMMs:
GAMM_predictions_2D_path="./datasets/2D_GAMM_predictions"
GAMM_predictions_LO_path="./datasets/LO_GAMM_predictions"
def get_normalized_l2(word, sense):
    preds_2D = pd.read_csv(os.path.join(GAMM_predictions_2D_path, f"{word}-{sense}.csv"))
    preds_LO = pd.read_csv(os.path.join(GAMM_predictions_LO_path, f"{word}-{sense}.csv"))
    #
    preds_2D = preds_2D.sort_values(by=['year', 'age'])
    preds_LO = preds_LO.sort_values(by=['year', 'age'])
    #
    deltas = preds_2D['probability']-preds_LO['probability']
    delta_from_mean_2D = preds_2D['probability'] - np.mean(preds_2D['probability'])
    delta_squared = deltas.apply(lambda x: x**2)
    delta_squared_from_mean_2D = delta_from_mean_2D.apply(lambda x: x**2)
    l2_distance_from_mean_2D = np.sum(delta_squared_from_mean_2D)
    l2_distance = np.sum(delta_squared)
    l2_normalized = l2_distance/l2_distance_from_mean_2D
    return l2_normalized 

processed_data['l2_distance_normalized'] = processed_data.apply(lambda x: get_normalized_l2(x['word'], x['sense']), axis=1)

# Add a column for prediction peak years:
def get_prediction_peak(word, sense):
    preds = pd.read_csv(os.path.join(GAMM_predictions_LO_path, f"{word}-{sense}.csv"))
    yearwise_means = preds.groupby('year').agg(mean_p=('probability', 'mean'))
    max_prob_year = yearwise_means[yearwise_means['mean_p'] == max(yearwise_means['mean_p'])].index.item()
    return max_prob_year

processed_data['peak_year'] = processed_data.apply(lambda x: get_prediction_peak(x['word'], x['sense']), axis=1)

# Remove tokenization artefacts that were found through manual inspection:
tokenization_artefacts = [
    'gay5',
    'guy9',
    'dial3',
    'organ3',
    'cap9',
]

wordsense_tags = processed_data['word']+processed_data['sense']
processed_data = processed_data[~wordsense_tags.apply(lambda x: x in tokenization_artefacts)]

processed_data.to_csv("collated_results.csv", index=False)
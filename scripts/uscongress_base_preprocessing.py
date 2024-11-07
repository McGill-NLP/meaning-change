import os
import string
import re 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

### Read in .csv:
print("Reading in .csv...")
usc_df = pd.read_csv("datasets/uscongress_linked.csv")

# Remove any cases of DOB being NAs: 
usc_df = usc_df[~usc_df['DOB'].isna()]

# Create columns for year of birth and year of speech, coded as ints:
print("Adding columns for year of birth, year of speech, and age at time of speech...")
usc_df['yob'] = usc_df['DOB'].apply(lambda x: int(x[0:4]))
usc_df['year'] = usc_df['date'].apply(lambda x: int(str(x)[0:4]))

# Create column for age at time of speech:
usc_df['age'] = usc_df['year'] - usc_df['yob']

# Make speakerIDs constant for each member, instead of changing with each session of congress:
# Have verified with some scratchpad code: there is no case of one speakerID mapping to more than one person!
usc_df['speakerid'] = usc_df['speakerid'].apply(lambda x: str(x)[-6:-1]) # See the Stanford US Congressional Record Codebook for more: https://stacks.stanford.edu/file/druid:md374tz9962/codebook_v4.pdf 

# Filtering for procedural speech using the Card et al. weights:
# Import weights as df:
weights = pd.read_table("assets/congress_procedural_weights.tsv", index_col=0)

# n-gram loader:
def load_ngrams(text):
    preprocessed = text.lower() # remove case 
    preprocessed = re.sub(r"[^\w\s]", "", preprocessed) # remove non-alphanumeric while leaving spaces for tokenization 
    unigrams = text.split()
    bigrams = ['_'.join([unigrams[i], unigrams[i+1]]) for i in range(len(unigrams)-1)]
    trigrams = ['_'.join([unigrams[i], unigrams[i+1], unigrams[i+2]]) for i in range(len(unigrams)-2)]
    return unigrams+bigrams+trigrams

# probability calculator:
def get_prob(ngram_list):
    filtered_ngrams = [x for x in ngram_list if x in weights.index] # Filter out n-grams with no weights in log-regression model; same as setting weights to zero
    collected_weights = weights.loc[filtered_ngrams]
    bias = weights.loc["__BIAS__"]
    prob = 1/(1+np.exp(-(np.sum(collected_weights) + bias))).item()
    return prob 

# Helper function: assign probability (of speech being procedural) of 0 if speech more than 400 characters, otherwise calculate probability of speech being procedural
def assign_procedural_prob(speech):
    if len(speech) > 400:
        return float(0)
    else:
        return get_prob(load_ngrams(speech))

# Create column with procedural speech probabilities so we can set our own filtering threshold later:
print("Adding procedural probabilities...")
usc_df['procedural_prob'] = usc_df['speech'].apply(lambda x: assign_procedural_prob(x))

### Write to .csv:
print("Writing to .csv...")
usc_df.to_csv("datasets/uscongress_base_preprocessed.csv", index=False)
print("All done!")
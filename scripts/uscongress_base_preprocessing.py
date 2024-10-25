import os
import string
import re 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

### Read in .csv:
usc_df = pd.read_csv("datasets/uscongress_linked.csv")

'''
Changing this pipeline for now -- will figure out how to move this code elsewhere later.

### Add columns for year of birth and year of speech.
usc_df['yob'] = usc_df['DOB'].apply(lambda x: int(x[0:4]))
usc_df['year'] = usc_df['date'].apply(lambda x: int(str(x)[0:4]))

### Add column for age at time of speech, also helps filter out bad entries where age < 25.
usc_df['age'] = usc_df['year'] - usc_df['yob']
usc_df = usc_df[usc_df['age']>=25]

### Adding columns for RANGES of year of speech and year of birth:
# Getting ranges of years:
# year_min = min(usc_df['year']) ## Saving as vars because huge df so multiple calls inefficient -- ends up being 1873
# year_max = max(usc_df['year']) ## same as above -- ends up being 2010
year_min = 1870 # based on actual min/max values from above, but round decade figures to allow for matching year ranges with year of birth ranges
year_max = 2010
year_slice = 10
year_bases = list(range(year_min, year_max, year_slice))
year_ranges = []
for i in range(len(year_bases)):
    if i==len(year_bases)-1:
        year_range = (year_bases[i], 2010)
    else:
        year_range = (year_bases[i], year_bases[i+1]-1)
    year_ranges.append(year_range)

year_range_array = np.array(sum(year_ranges, ()))

# Getting ranges of YOBs:
# yob_min = min(usc_df['yob']) ## Saving as vars because huge df so multiple calls inefficient -- ends up being 1771
# yob_max = max(usc_df['yob']) ## Same as above -- ends up being 1981
yob_min = 1770 # based on actual min/max values from above, but round decade figures to allow for matching year ranges with year of birth ranges
yob_max = 1990
yob_slice = 10
yob_bases = list(range(yob_min, yob_max, yob_slice))
yob_ranges = []
for i in range(len(yob_bases)):
    if i==len(yob_bases)-1:
        yob_range = (yob_bases[i], max(usc_df['yob']))
    else:
        yob_range = (yob_bases[i], yob_bases[i+1]-1)
    yob_ranges.append(yob_range)

yob_range_array = np.array(sum(yob_ranges, ()))

# Helper function:
def find_range(year, tuple_list, array):
    idx_in_array = np.argmin(abs(array-year))
    idx_in_tuple_list = idx_in_array//2
    return tuple_list[idx_in_tuple_list]

# Creating columns:
usc_df['year_range'] = usc_df['year'].apply(lambda x: find_range(x, year_ranges, year_range_array))
usc_df['yob_range'] = usc_df['yob'].apply(lambda x: find_range(x, yob_ranges, yob_range_array))

### Preprocessing: Removing case and punctuation from speech text
usc_df['speech'] = usc_df['speech'].apply(lambda x: re.sub(r"[^\w\s]", "", x.lower()) )

### Creating plots:
usc_df['speech_length'] = usc_df['speech'].apply(lambda x: len(x.split()))
grouped_df = usc_df.groupby(['year_range', 'yob_range'])['speech_length'].sum().reset_index()
grouped_df['year_range'] = grouped_df['year_range'].apply(lambda x: str(x))
grouped_df['yob_range'] = grouped_df['yob_range'].apply(lambda x: str(x))
g = sns.FacetGrid(grouped_df, col='year_range', col_wrap=4, sharey=False, sharex=False)
g.map(sns.barplot, 'yob_range', 'speech_length', order=sorted(grouped_df['yob_range'].unique()))
for ax in g.axes:
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.tight_layout()
g.savefig("../plots/congress_facet.png")

'''
# Create columns for year of birth and year of speech, coded as ints:
usc_df['yob'] = usc_df['DOB'].apply(lambda x: int(x[0:4]))
usc_df['year'] = usc_df['date'].apply(lambda x: int(str(x)[0:4]))

# Create column for age at time of speech:
usc_df['age'] = usc_df['year'] - usc_df['yob']

# Filtering for procedural speech using Dallas' weights:
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
usc_df['procedural_prob'] = usc_df['speech'].apply(lambda x: assign_procedural_prob(x))

### Write to .csv:
usc_df.to_csv("datasets/uscongress_base_preprocessed.csv", index=False)
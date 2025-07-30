from tqdm import tqdm
import os
import numpy as np
import pandas as pd 
import re
from utils.data import yaml_to_df, summarize_yaml_df

'''
PATH_TO_HEIN_BOUND: Path to the unzipped hein-bound folder
PATH_TO_LEG_HIST: Path to the .yaml file containing historical US Congress legislator info
PATH_TO_LEG_CURRENT: Path to the .yaml file containing current US Congress legislator info
'''

path_to_hein_bound = os.getenv("PATH_TO_HEIN_BOUND")

### Sticking Together DOB Files:
path_to_leg_hist = os.getenv("PATH_TO_LEG_HIST")
path_to_leg_current = os.getenv("PATH_TO_LEG_CURRENT")
leg_hist_df = yaml_to_df(path_to_leg_hist)
leg_current_df = yaml_to_df(path_to_leg_current)
leg_all_df = pd.concat((leg_current_df, leg_hist_df))
leg_summarized_df = summarize_yaml_df(leg_all_df)
print("DOB files compiled!")

### Sticking together all Mapping Files:
path_to_speakermaps = os.path.join(path_to_hein_bound, 'speakermaps/')
speakermap_files = sorted(os.listdir(path_to_speakermaps))
speakermap_df_list = []
for filename in tqdm(speakermap_files):
    speakermap_df = pd.read_csv(path_to_speakermaps+filename, sep="|")
    speakermap_df_list.append(speakermap_df)

speakermaps_compiled_df = pd.concat(speakermap_df_list)
print("Speakermap files compiled!")

### Sticking together all Description files (these contain speech dates):
path_to_descriptions = os.path.join(path_to_hein_bound, 'descriptions/')
descriptions_files = sorted(os.listdir(path_to_descriptions))
descriptions_df_list = []
for filename in tqdm(descriptions_files):
    description_df = pd.read_csv(path_to_descriptions+filename, sep="|", encoding='latin1')
    descriptions_df_list.append(description_df)

descriptions_compiled_df = pd.concat(descriptions_df_list)
print("Description files (dates of speeches) compiled!")

### Matching Descriptions files and Speakermaps:

# Filter for descriptions with corresponding speakermaps:
descriptions_filtered = descriptions_compiled_df[descriptions_compiled_df['speech_id'].isin(speakermaps_compiled_df['speech_id'])]
print("Description files filtered for those with matching speakermaps!")

# Get dates from description files into speakermaps:
descriptions_filtered = descriptions_filtered.set_index("speech_id") # to make next step easier
speakermaps_compiled_df['date'] = speakermaps_compiled_df['speech_id'].apply(lambda x: descriptions_filtered.loc[x]['date'])
print("Dates of speeches attached to speakermaps!")

### Compile first name / last name / state in each df for matching
chamber_conversion = {"sen":"S", "rep":"H"}

speakermaps_compiled_df['identifiers'] = speakermaps_compiled_df['firstname']+" "+speakermaps_compiled_df['lastname']+" "+speakermaps_compiled_df['chamber']+" "+speakermaps_compiled_df['state']
speakermaps_compiled_df['identifiers'] = speakermaps_compiled_df['identifiers'].str.lower()

leg_summarized_df['identifiers'] = leg_summarized_df['name.first']+" "+leg_summarized_df['name.last']+" "+leg_summarized_df['terms.type'].apply(lambda x: chamber_conversion[x])+" "+leg_summarized_df['terms.state']
leg_summarized_df['identifiers'] = leg_summarized_df['identifiers'].str.lower()
print("Identifier tags prepared for speakermap / DOB file matching!")

# Removing duplicates in the DOB dataset
unique_name_counts = np.unique(leg_summarized_df['identifiers'], return_counts = True)
duplicate_names_indices = np.where(unique_name_counts[1] > 1)[0]
duplicate_names = unique_name_counts[0][duplicate_names_indices]
leg_summarized_df = leg_summarized_df[~leg_summarized_df['identifiers'].isin(duplicate_names)]
print("People with identical identifiers removed!")

# Now we filter for items in the Speakermap dataset that have identifier strings present in the DOB dataset
speakermaps_filtered = speakermaps_compiled_df[speakermaps_compiled_df['identifiers'].isin(leg_summarized_df['identifiers'])]
print("Speakermaps filtered for those with viable identifiers!")

# Set 'identifiers' column as index to help us with matching:
leg_summarized_df = leg_summarized_df.set_index('identifiers')

# Link it all up!
speakermaps_filtered['DOB'] = speakermaps_filtered['identifiers'].apply(lambda x: leg_summarized_df.loc[x]['bio.birthday'])
print("Dates of birth linked to speakermaps!")

# Drop 'identifiers' column; we don't need it anymore
speakermaps_filtered = speakermaps_filtered.drop(columns=['identifiers'])

### Speeches
# Stick all Speeches together:
path_to_speeches = os.path.join(path_to_hein_bound, 'speeches/')
speech_files = sorted(os.listdir(path_to_speeches))
speeches_df_list = []
for filename in tqdm(speech_files):
    speech_df = pd.read_csv(path_to_speeches+filename, sep=r'^([^\|]+)\|', encoding="latin1", engine="python")
    speeches_df_list.append(speech_df)

speeches_compiled_df = pd.concat(speeches_df_list)
print("Speech files compiled!")


# Filter speeches for those that can be connected to speakermaps
speeches_filtered = speeches_compiled_df[speeches_compiled_df['speech_id'].isin(speakermaps_filtered['speech_id'])]
print("Speeches filtered for those that can be connected to speakermaps!")

# Make speech_id the index to help with next step
speeches_filtered = speeches_filtered.set_index("speech_id")

# Put it all together!
speakermaps_filtered['speech'] = speakermaps_filtered['speech_id'].apply(lambda x: speeches_filtered.loc[x]['speech'])
print("Speeches linked to speakermaps: everything's ready!")

# Write to .csv:
csv_path = "datasets/uscongress_linked.csv"
speakermaps_filtered.to_csv(csv_path, index=False)
print(".csv file written, all done!")

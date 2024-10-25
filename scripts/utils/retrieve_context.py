import pandas as pd
import numpy as np 
import os
from transformers import AutoTokenizer

print("Heads-up: this function may take a while to import, because it involves reading in the original corpus in order to retrieve speeches.")

# These act as overall settings for the function, that are likely too stable to warrant being function arguments. But they can be easily changed if desired:
corpus_path = './datasets/uscongress_base_preprocessed.csv'
procedural_prob_threshold = 0.5
corpus_age_min = 25
corpus = pd.read_csv(corpus_path)
corpus = corpus[corpus['age']>=corpus_age_min] # Any age entries below 25 are going to be errors -- remove them
corpus = corpus[corpus['procedural_prob']<procedural_prob_threshold]
clustering_results_dir = './clusters/'
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased') # This needs to be the same tokenizer as was used during the substitution generation step
columns_to_return = ['speakerid', 'speech_id', 'lastname', 'firstname', 'year', 'yob', 'age', 'left_context', 'right_context']

def get_lr_context_df(word, context_window=50):
    word_cluster_results = pd.read_csv(os.path.join(clustering_results_dir, f"{word}_cluster_counts.csv"))
    #
    relevant_speech_ids = word_cluster_results['speech_id']
    relevant_speeches = corpus[corpus['speech_id'].isin(relevant_speech_ids)]
    #
    target_token_ids = tokenizer.encode(word, add_special_tokens=False)
    with pd.option_context('mode.chained_assignment', None):
        relevant_speeches['speech_tokenized'] = relevant_speeches['speech'].apply(lambda x: tokenizer.encode(x, add_special_tokens=False))
    #
    def get_lr_contexts_from_tokenized_speech(input_ids, context_window=50):
        target_indices = [i for i in range(len(input_ids) - len(target_token_ids) + 1) if (input_ids[i:i + len(target_token_ids)] == target_token_ids)]
        left_context_tokens = [input_ids[target_index-context_window:target_index] for target_index in target_indices]
        right_context_tokens = [input_ids[target_index+len(target_token_ids):target_index+1+context_window] for target_index in target_indices]
        left_contexts = [tokenizer.decode(tokens) for tokens in left_context_tokens]
        right_contexts = [tokenizer.decode(tokens) for tokens in right_context_tokens]
        return left_contexts, right_contexts
    #
    lr_contexts_zipped = relevant_speeches['speech_tokenized'].apply(lambda x: get_lr_contexts_from_tokenized_speech(x, context_window=context_window))
    with pd.option_context('mode.chained_assignment', None):
        relevant_speeches['left_context'] = lr_contexts_zipped.apply(lambda x: x[0])
        relevant_speeches['right_context'] = lr_contexts_zipped.apply(lambda x: x[1])
    #
    relevant_speeches_expanded = relevant_speeches.explode(['left_context', 'right_context']).reset_index(drop=True)
    context_info = relevant_speeches_expanded[['speech_id', 'left_context', 'right_context']]
    merged = pd.merge(context_info, word_cluster_results, how='left', on='speech_id')
    merged.rename(columns=lambda x: x + '_agg' if x.startswith('cluster_') or x=='n' else x, inplace=True)
    return merged


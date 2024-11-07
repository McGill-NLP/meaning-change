import re 
import json 
import itertools 
import networkx as nx 
import community as community_louvain
import pandas as pd 
import numpy as np 
import argparse 
import os 
from tqdm import tqdm 
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus-path', type=str, default='./datasets/uscongress_base_preprocessed.csv')
    parser.add_argument('--procedural-prob', type=int, default=0.5)
    parser.add_argument('--substitution-directory', type=str, default='./substitutions/')
    parser.add_argument('--wordlist-path', type=str, default='./assets/wordlist.txt')
    parser.add_argument('--clustering-threshold', type=int, default=0)
    parser.add_argument('--same-word-ok', type=bool, default=False)
    parser.add_argument('--use-log-probs', type=bool, default=True)
    parser.add_argument('--output-directory', type=str, default='./clusters/')
    args = parser.parse_args()
    run_clustering(args)


def get_clusters(word: str, substitution_directory: str, threshold=0, same_word_ok=False, use_log_probs=True):
    data = []
    with open(os.path.join(substitution_directory, f'{word}.jsonl')) as file:
        for line in file:
            data.append(json.loads(line))
    #
    G = nx.Graph()
    node_frequencies = {}
    #
    co_occurrence_threshold = threshold * len(data)
    # Add nodes and edges based on the substitution co-occurrences in the data
    for item in data:
        substitution_lists = item['substitutions']
        log_probs_lists = item['log_probs']
        
        for idx, substitution_list in enumerate(substitution_lists):
            log_probs = log_probs_lists[idx]
            
            # Update node frequencies
            for substitution in substitution_list:
                if not same_word_ok:
                    if substitution != word:
                        node_frequencies[substitution] = node_frequencies.get(substitution, 0) + 1
                else:
                    node_frequencies[substitution] = node_frequencies.get(substitution, 0) + 1
            
            # Add edges:
            substitution_list = [x for x in substitution_list if x != word] if not same_word_ok else substitution_list
            for sub_pair in itertools.combinations(substitution_list, 2):
                if G.has_edge(*sub_pair):
                    G[sub_pair[0]][sub_pair[1]]['weight'] += 1
                else:
                    G.add_edge(sub_pair[0], sub_pair[1], weight=1)
                
                # Adjust weight based on log probabilities if enabled
                if use_log_probs:
                    i, j = substitution_list.index(sub_pair[0]), substitution_list.index(sub_pair[1])
                    prob_adjustment = (np.exp(log_probs[i]) + np.exp(log_probs[j])) / 2
                    G[sub_pair[0]][sub_pair[1]]['weight'] *= prob_adjustment
    
    # Remove edges with weights below the threshold
    for u, v, G_data in list(G.edges(data=True)):
        if G_data['weight'] < co_occurrence_threshold:
            G.remove_edge(u, v)
    
    # Remove isolated nodes that may have been created by edge removal
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    # Apply the Louvain method for community detection
    partition = community_louvain.best_partition(G)
    
    sorted_nodes = sorted(partition.keys(), key=lambda node: node_frequencies[node], reverse=True)
    # Group the nodes by their community
    clusters = {}
    for node in sorted_nodes:
        comm_id = partition[node]
        clusters.setdefault(comm_id, []).append(node)
    
    return clusters


def get_clusterwise_counts_optimized(substitutions: pd.Series, clusters: dict):
    # Initialize a dictionary to hold the counts for each cluster
    clusterwise_counts = {key: [] for key in clusters.keys()}
    
    # Iterate over each row's substitutions and count occurrences per cluster
    for subs in substitutions:
        flattened_subs = [item for sublist in subs for item in sublist]
        for cluster, words in clusters.items():
            # Use set intersection for efficient count of occurrences
            count = sum(flattened_subs.count(word) for word in words)
            clusterwise_counts[cluster].append(count)
    
    return clusterwise_counts

def get_clusterwise_count_df_optimized(word_df: pd.DataFrame, clusters: dict):
    # Get the cluster-wise counts for the entire DataFrame at once
    cluster_counts = get_clusterwise_counts_optimized(word_df['substitutions'], clusters)
    
    # Add the counts back to the DataFrame as new columns
    for cluster, counts in cluster_counts.items():
        word_df[f'cluster_{cluster}'] = counts
    
    return word_df

def write_cluster_count_info(word_df_processed, clusters, word, output_directory):
    total_count = np.sum([np.sum(word_df_processed[f"cluster_{i}"]) for i in range(len(clusters.keys()))])
    with open(os.path.join(output_directory, f'{word}_cluster_info.txt'), 'w') as file:
        for key in clusters.keys():
            raw_count = np.sum(word_df_processed[f'cluster_{key}'])
            file.write(f"Cluster {key}:\n")
            file.write(f"Examples: {clusters[key][0:7]}\n")
            file.write(f"Raw count: {raw_count}\n")
            file.write(f"Proportion: {raw_count/total_count}\n\n")
    
def get_merged_word_df(word_df_processed, base_corpus):
    cluster_columns = [x for x in word_df_processed.columns if 'cluster' in x]
    word_df_processed['n'] = np.sum(word_df_processed[cluster_columns], axis=1)
    for cluster in cluster_columns:
        word_df_processed[f'{cluster}_p'] = word_df_processed[cluster]/word_df_processed['n']
    
    cluster_columns = [x for x in word_df_processed.columns if 'cluster' in x]
    
    word_speeches = base_corpus[base_corpus['speech_id'].isin(word_df_processed['speech_id'].tolist())]
    word_speeches_merged = word_speeches.merge(word_df_processed[cluster_columns+['speech_id', 'n']], how='outer', on=['speech_id'])
    word_speeches_merged = word_speeches_merged.drop(columns=['speech'])
    return word_speeches_merged

'''
def get_lineplots(merged_word_df, word, word_folder):
    cluster_prop_columns = [x for x in merged_word_df.columns if '_p' in x]
    plot_data = merged_word_df[['age', 'yob', 'year']+cluster_prop_columns]
    predictor_names = {'age': 'Age', 'yob': 'Year of Birth', 'year': 'Year of Speech'}
    # By Age, YOB and Year:
    for predictor in ['age', 'yob', 'year']:
        plot_data_grouped = plot_data.groupby(predictor).mean().reset_index()
        plt.figure(figsize=(12,8))
        for cluster_col in cluster_prop_columns:
            plt.plot(plot_data_grouped[predictor], plot_data_grouped[cluster_col], label=cluster_col)
        
        plt.xlabel(predictor_names[predictor])
        plt.ylabel('Mean Proportion of Cluster-wise Word Sense Uses')
        plt.title(f'Proportion of Cluster-wise Word Sense Uses by {predictor_names[predictor]}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(word_folder, f"{word}_{predictor}_lineplot.png"))
        plt.clf()


'''

def run_clustering(args):
    corpus_filepath = args.corpus_path
    procedural_prob_cutoff = args.procedural_prob
    substitution_directory = args.substitution_directory
    wordlist_path = args.wordlist_path
    clustering_threshold = args.clustering_threshold
    same_word_ok = args.same_word_ok
    use_log_probs = args.use_log_probs
    output_directory = args.output_directory
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    #
    all_speeches = pd.read_csv(corpus_filepath)
    all_speeches = all_speeches[all_speeches['procedural_prob']<procedural_prob_cutoff]
    #
    word_list = []
    with open(wordlist_path, 'r') as file:
        for line in file:
            word_list.append(line.strip())
    #
    for word in tqdm(word_list):
        #word_folder = os.path.join(output_directory, word)
        #if not os.path.exists(word_folder):
        #    os.mkdir(word_folder)
        #
        word_data = []
        with open(os.path.join(substitution_directory, f"{word}.jsonl"), 'r') as file:
            for line in file:
                word_data.append(json.loads(line))
        #
        word_df = pd.DataFrame(word_data)
        clusters = get_clusters(word, substitution_directory=substitution_directory, threshold=clustering_threshold, same_word_ok=same_word_ok, use_log_probs=use_log_probs)
        word_df_processed = get_clusterwise_count_df_optimized(word_df, clusters)
        write_cluster_count_info(word_df_processed, clusters, word, output_directory)
        merged_word_df = get_merged_word_df(word_df_processed, all_speeches)
        merged_word_df.to_csv(os.path.join(output_directory, f"{word}_cluster_counts.csv"), index=False)
        # get_lineplots(merged_word_df, word, word_folder)

if __name__=='__main__':
    main()
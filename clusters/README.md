Files in this folder are of two types:
- `{word}_cluster_counts.csv`
- `{word}_cluster_info.txt`

The `cluster_counts` files contain information on, for each speech containing the word, how many predicted replacement tokens belong to each replacement cluster (`n` = total number of replacement tokens generated, `cluster_X` = number of replacement tokens belonging to cluster number X).

The `cluster_info` files contain information on, for each replacement cluster for a given word, what the top replacements in the cluster are, and the proportion of the word's generated replacements that belong to the cluster overall.  
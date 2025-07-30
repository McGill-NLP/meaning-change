# Semantic Change in Adults is Not Primarily a Generational Phenomenon

*Gaurav Kamath*, *Michelle Yang*, *Siva Reddy*, *Morgan Sonderegger*, *Dallas Card*

This repository contains code and data for the 2025 paper "Semantic Change in Adults is Not Primarily a Generational Phenomenon", published in *Proceedings of the National Academy of Sciences* (*PNAS*).

You can access the paper here: https://www.pnas.org/doi/10.1073/pnas.2426815122.

## Contents

Below are the structure and contents of this repository.* 

- `a-optimization_results`: Results of alpha-optimization for each sense studied of each word.
- `assets`: Extra files used for some of the data processing.
- `clusters`: Word-wise results of the replacement clustering algorithm, used to estimate word sense likelihood.
- `datasets`: As uploaded to this repository, only datasets containing GAMM predictions; but we recommend adding the preprocessed corpus files here if you plan to reproduce our experiments.
- `plots`: All plots generated as part of this study.
- `scripts`: Python and R scripts used to process the data, fit models, and generate plots.
- `substitutions`: The target word replacement data used to generate replacement clusters for word sense induction.
- `collated_results.csv`: CSV file containing the final results of our analysis by each word sense studied.
- `meta-analysis.rds`: RDS file that stores the Bayesian Meta-Analysis model we fit to estimate an average age effect.
- `requirements.txt`: Python libraries required for the Python scripts to run.
- `r_env.yml`: R libraries (using a Conda environment) required for the R scripts to run.

*Note that the original Stanford Congressional Record Dataset is not included in this repository, as it is far too large. The original data can be accessed [here](https://data.stanford.edu/congress_text) (download `hein-bound.zip`), and then processed using the Python scripts in our `scripts` folder. To do so, you will also need to download the `legislators-current.yaml` and `legislators-historical.yaml` files from [here](https://github.com/unitedstates/congress-legislators). 

Furthermore, note that the following files (related to high-frequency words) are also not included in this repository on account of being too large: 

- `substitutions/call.jsonl`
- `substitutions/high.jsonl`
- `clusters/record_cluster_counts.csv`
- `substitutions/record.jsonl`
- `substitutions/security.jsonl`  

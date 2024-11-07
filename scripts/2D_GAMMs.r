library(glue)
library(mgcv)
library(readr)
library(tidyverse)
library(lme4)
library(stringr)
library(stats4)
library(purrr)

word <- Sys.getenv("WORD")
change_threshold <- as.numeric(Sys.getenv('CHANGE_THRESHOLD'))
window_size <- as.numeric(Sys.getenv("WINDOW_SIZE"))
sample_size <- as.numeric(Sys.getenv("SAMPLE_SIZE"))
random_seed <- as.numeric(Sys.getenv("RANDOM_SEED"))
saved_model_dir <- Sys.getenv("SAVED_MODEL_DIR")
if (!file.exists(saved_model_dir)) {
  dir.create(saved_model_dir)
}

get_high_change_senses_word_tibble <- function(word_tibble, window_size = 10, threshold = 0.3) {
    # Identify the cluster probability columns
    cluster_p_columns <- grep("^cluster_.*_p$", colnames(word_tibble), value = TRUE)
    high_change_columns <- c()
    
    # Iterate over the identified columns
    for (column in cluster_p_columns) {
        years <- unique(word_tibble$year)
        mean_props <- c()
        
        # Calculate mean proportions over sliding windows
        for (start_year in years) {
            end_year <- start_year + window_size
            if (end_year <= max(years)) {
                window_data <- word_tibble %>%
                    filter(year >= start_year & year < end_year)
                mean_prop <- mean(window_data[[column]], na.rm = TRUE)
                mean_props <- c(mean_props, mean_prop)
            } else {
                break
            }
        }
        
        # Check if the difference between max and min exceeds the threshold
        max_prop <- max(mean_props, na.rm = TRUE)
        min_prop <- min(mean_props, na.rm = TRUE)
        if (max_prop - min_prop > threshold) {
            high_change_columns <- c(high_change_columns, column)
        }
    }
    
    # Extract corresponding cluster columns to include based on high-change probability columns
    high_change_cluster_columns <- high_change_columns %>%
        gsub("_p$", "", .) %>%
        unique()
    
    # Determine columns to exclude
    cluster_columns_to_exclude <- setdiff(grep("^cluster_\\d+$", colnames(word_tibble), value = TRUE), high_change_cluster_columns)
    cluster_p_columns_to_exclude <- setdiff(cluster_p_columns, high_change_columns)
    columns_to_exclude <- c(cluster_columns_to_exclude, cluster_p_columns_to_exclude)
    
    # Select all columns except those marked for exclusion
    high_change_senses_word_tibble <- word_tibble %>%
        select(-all_of(columns_to_exclude))
    
    return(high_change_senses_word_tibble)
}



## Pivot:
pivot_custom <- function(high_change_senses_word_tibble){
    high_change_senses_long <- high_change_senses_word_tibble %>%
        select(-matches("^cluster_\\d+$")) %>% # Super roundabout, but we exclude the `cluster_<number>` columns here to recalculate them after pivoting
        pivot_longer(
            cols = starts_with("cluster_"),  # Select columns that start with "cluster_"
            names_to = "cluster_number",     # Name of the new column for the cluster numbers # nolint: line_length_linter.
            names_prefix = "cluster_",       # Remove the prefix "cluster_" from the original column names
            values_to = "cluster_p"          # Name of the new column for the cluster probabilities # nolint: line_length_linter.
        ) %>%
        mutate(cluster_number = factor(as.numeric(str_extract(cluster_number, "\\d+")))) %>%
        mutate(n_cluster = round(cluster_p*n))
    #
    return(high_change_senses_long)
}

get_sample <- function(tibble, sample_size, random_seed){
  set.seed(random_seed)
  sample <- sample_n(tibble, sample_size)
  return(sample)
}

word_tibble <- read.csv(glue("./clusters/{word}_cluster_counts.csv"))
high_change_senses_word_tibble <- get_high_change_senses_word_tibble(word_tibble, window_size=window_size, threshold=change_threshold)
n_speeches <- nrow(high_change_senses_word_tibble)
if(n_speeches > sample_size){
  data_for_modelling <- get_sample(high_change_senses_word_tibble, sample_size, random_seed)
} else{
  data_for_modelling <- high_change_senses_word_tibble
}

data_long <- pivot_custom(data_for_modelling)

for (cluster in unique(data_long$cluster_number)){
  filename <- glue("{saved_model_dir}/{word}-{cluster}-2D_GAMM.rds")
  models_fitted <- list.files(saved_model_dir)
  if (!filename %in% models_fitted)
  {
    print(glue("Cluster {cluster}"))
    # Index to one sense:
    word_sense_indexed <- data_long %>% filter(cluster_number == cluster)
    word_sense_indexed$speakerid <- factor(word_sense_indexed$speakerid)
    #
    print(glue("Starting to fit full model at {Sys.time()}"))
    m_full_RE <- bam(cbind(n_cluster, n - n_cluster) ~ ti(age) + ti(year) + ti(age, year) + s(speakerid, bs = 're'), family = binomial, data = word_sense_indexed, discrete = TRUE)
    print(glue("Full model fitted at {Sys.time()}"))
    print("Now saving full model...")
    saveRDS(m_full_RE, file=glue("{saved_model_dir}/{word}-{cluster}-2D_GAMM.rds"))
    print("Saved!")
    rm(m_full_RE)
    }
}



    


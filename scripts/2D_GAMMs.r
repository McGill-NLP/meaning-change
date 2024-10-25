library(glue)
library(mgcv)
library(readr)
library(tidyverse)
library(lme4)
library(stringr)
library(stats4)
library(purrr)

word <- Sys.getenv("WORD")
window_size <- as.numeric(Sys.getenv("WINDOW_SIZE"))
prominence_threshold <- as.numeric(Sys.getenv("PROMINENCE_THRESHOLD"))
sample_size <- as.numeric(Sys.getenv("SAMPLE_SIZE"))
speaker_threshold <- as.numeric(Sys.getenv("SPEAKER_THRESHOLD"))
random_seed <- as.numeric(Sys.getenv("RANDOM_SEED"))
saved_model_dir <- Sys.getenv("SAVED_MODEL_DIR")
if (!file.exists(saved_model_dir)) {
  dir.create(saved_model_dir)
}

get_prominent_senses <- function(word_tibble, window_size=20, threshold=0.2) {
    #
    cluster_columns <- names(word_tibble)[str_detect(names(word_tibble), "^cluster_\\d+$")]
    
    max_proportion_in_window <- function(column) {
        max_proportion <- 0
        years <- word_tibble$year
        for (start_year in unique(years)) {
            end_year <- start_year + window_size
            window_data <- word_tibble %>%
            filter(year >= start_year & year < end_year)
            proportion <- sum(window_data[[column]], na.rm = TRUE) / sum(window_data$n, na.rm = TRUE)
            if (proportion > max_proportion) {
            max_proportion <- proportion
            }
        }
        return(max_proportion)
        }
    #
    # Determine which columns to exclude
    columns_to_exclude <- cluster_columns %>%
        map_lgl(function(column) {
            max_proportion <- max_proportion_in_window(column)
            max_proportion < threshold
        })
    
    # Exclude low-frequency columns
    excluded_cluster_columns <- cluster_columns[columns_to_exclude]
    excluded_p_columns <- purrr::map_chr(excluded_cluster_columns, ~paste0(.x, "_p"))
    final_columns_to_exclude <- c(excluded_cluster_columns, excluded_p_columns)
    prominent_senses_word_tibble <- word_tibble %>%
        select(-all_of(final_columns_to_exclude))
    #
    # print("Data filtered for prominent senses!")
    return(prominent_senses_word_tibble)
}

## Pivot:
pivot_custom <- function(prominent_senses_word_tibble){
    prominent_senses_long <- prominent_senses_word_tibble %>%
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
    return(prominent_senses_long)
}

filter_by_speaker_uses_of_word <- function(word_tibble, n_threshold=1){
    prolific_speakers <- word_tibble %>% count(speakerid) %>% filter(n > n_threshold) %>% pull(speakerid)
    word_tibble_sub <- filter(word_tibble, speakerid %in% prolific_speakers)
    return(word_tibble_sub)
}

get_sample <- function(tibble, sample_size, random_seed){
  set.seed(random_seed)
  sample <- sample_n(tibble, sample_size)
  return(sample)
}

word_tibble <- read.csv(glue("./clusters/{word}_cluster_counts.csv"))
prominent_senses_word_tibble <- get_prominent_senses(word_tibble, window_size=window_size, threshold=prominence_threshold)
n_speeches <- nrow(prominent_senses_word_tibble)
if(n_speeches > sample_size){
  data_for_modelling <- get_sample(prominent_senses_word_tibble, sample_size, random_seed)
} else{
  data_for_modelling <- prominent_senses_word_tibble
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



    


library(glue)
library(mgcv)
library(readr)
library(tidyverse)
library(lme4)
library(stringr)
library(stats4)
library(purrr)
library(jsonlite)
library(tibble)

word <- Sys.getenv("WORD")
# window_size <- as.numeric(Sys.getenv("WINDOW_SIZE"))
# prominence_threshold <- as.numeric(Sys.getenv("PROMINENCE_THRESHOLD"))
sample_size <- as.numeric(Sys.getenv("SAMPLE_SIZE"))
# speaker_threshold <- as.numeric(Sys.getenv("SPEAKER_THRESHOLD"))
random_seed <- as.numeric(Sys.getenv("RANDOM_SEED"))
saved_model_dir <- Sys.getenv("SAVED_MODEL_DIR")
if (!file.exists(saved_model_dir)) {
  dir.create(saved_model_dir)
}

word_filepath <- "./assets/wordlist.txt"
words <- str_trim(readLines(word_filepath))

a_optimization_results_directory <- "./a-optimization_results"
a_optimization_results <- list()
for (i in words) {
  json_file <- glue("a-optim-{i}.json")
  full_filepath <- file.path(a_optimization_results_directory, json_file)
  
  if (file.exists(full_filepath)) {
    json_data <- fromJSON(full_filepath)
    a_optimization_results[[i]] <- json_data
  }
}

get_a_optimized_senses <- function(word, a_optimization_results) {
  if (!word %in% names(a_optimization_results)) {
    return(NULL)
  } else {
    word_results <- as_tibble(a_optimization_results[[word]])
    senses_a_optimized <- sapply(word_results$sense, function(x){x})
    return(senses_a_optimized)
  }
}

get_a_optimized_senses_tibble <- function(word_tibble, a_optimized_senses){
    output_word_tibble <- word_tibble %>% select(-starts_with('cluster'))
    optimized_cluster_names <- sapply(a_optimized_senses, function(x){glue("cluster_{x}")})
    optimized_p_cluster_names <- sapply(a_optimized_senses, function(x){glue("cluster_{x}_p")})
    all_optimized_cluster_names <- c(optimized_cluster_names, optimized_p_cluster_names)
    for (cluster_name in all_optimized_cluster_names){
      output_word_tibble[cluster_name] = word_tibble[cluster_name]
    }
    return(output_word_tibble)
}

get_optimized_a_coef <- function(word, target_sense, a_optimization_results){
  word_results <- as_tibble(a_optimization_results[[word]])
  a_estimate <- word_results %>% filter(sense == target_sense) %>% .$a_estimate %>% .[[1]]
  return(a_estimate)
}

## Pivot:
pivot_custom <- function(a_optimized_senses_word_tibble){
    a_optimized_senses_long <- a_optimized_senses_word_tibble %>%
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
    return(a_optimized_senses_long)
}

get_sample <- function(tibble, sample_size, random_seed){
  set.seed(random_seed)
  sample <- sample_n(tibble, sample_size)
  return(sample)
}

word_tibble <- read.csv(glue("./clusters/{word}_cluster_counts.csv"))
print(word)
a_optimized_senses <- get_a_optimized_senses(word, a_optimization_results)
print(a_optimized_senses)
a_optimized_senses_word_tibble <- get_a_optimized_senses_tibble(word_tibble, a_optimized_senses)
print(dim(a_optimized_senses_word_tibble))
n_speeches <- nrow(a_optimized_senses_word_tibble)
if(n_speeches > sample_size){
  data_for_modelling <- get_sample(a_optimized_senses_word_tibble, sample_size, random_seed)
} else{
  data_for_modelling <- a_optimized_senses_word_tibble
}

data_long <- pivot_custom(data_for_modelling)

for (cluster in unique(data_long$cluster_number)){
  filename <- "{word}-{cluster}-LO_GAMM.rds"
  full_filename <- glue("{saved_model_dir}/{filename}")
  models_fitted <- list.files(saved_model_dir)
  if (!filename %in% models_fitted)
  {
    print(glue("Cluster {cluster}"))
    # Get optimized a value:
    a_coef <- get_optimized_a_coef(word, cluster, a_optimization_results)
    # Index to one sense:
    word_sense_indexed <- data_long %>% filter(cluster_number == cluster)
    word_sense_indexed$speakerid <- factor(word_sense_indexed$speakerid)
    # Apply linear offset:
    word_sense_indexed <- word_sense_indexed %>% mutate(time = year-(a_coef*age))
    #
    print(glue("Starting to fit full model at {Sys.time()}"))
    m_full_RE <- bam(cbind(n_cluster, n - n_cluster) ~ s(time) + s(speakerid, bs='re'), family = binomial, data = word_sense_indexed, discrete = TRUE)
    print(glue("Full model fitted at {Sys.time()}"))
    print("Now saving full model...")
    saveRDS(m_full_RE, file=full_filename)
    print("Saved!")
    rm(m_full_RE)
    }
}



    


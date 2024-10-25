library(glue)
library(mgcv)
library(readr)
library(tidyverse)
library(lme4)
library(stringr)
library(stats4)
library(jsonlite)
library(tibble)

word <- Sys.getenv('WORD')
change_threshold <- as.numeric(Sys.getenv('CHANGE_THRESHOLD'))
window_size <- as.numeric(Sys.getenv('WINDOW_SIZE'))
speaker_threshold <- as.numeric(Sys.getenv("SPEAKER_THRESHOLD"))
output_directory <- './a-optimization_results'
if (!dir.exists(output_directory)){
    dir.create(output_directory)}

compute_fREML <- function(a, data) {
  # Update 'time' based on 'a'
  data$time <- data$year - (a * data$age)
  
  # Fit the model
  model <- bam(cbind(n_cluster, n - n_cluster) ~ s(time) + s(speakerid, bs='re'), 
               family = binomial, data = data, discrete = TRUE)
  
  # Get model summary:
  model_summary <- summary(model, re.test=FALSE) 
  # Return the fREML value
  return(as.numeric(model_summary$sp.criterion))
}

filter_down <- function(df, threshold=2000){for (i in seq(0, length(unique(df$speakerid)))){
  prolific_speakers <- df %>% count(speakerid) %>% filter(n > i) %>% pull(speakerid)
  filtered <- df %>% filter(speakerid %in% prolific_speakers)
  n_unique_speakers <- length(unique(filtered$speakerid))
  if (n_unique_speakers < threshold){
    return(filtered)
    break
  }
}
}

get_data <- function(word, cluster, threshold=2000){
  word_df <- read.csv(glue("clusters/{word}_cluster_counts.csv"))
  ## turn into long format
  word_long <- word_df %>%
  select(-matches("^cluster_\\d+$")) %>%
  pivot_longer(
      cols = starts_with("cluster_"),  # Select columns that start with "cluster_"
      names_to = "cluster_number",     # Name of the new column for the cluster numbers # nolint: line_length_linter.
      names_prefix = "cluster_",       # Remove the prefix "cluster_" from the original column names
      values_to = "cluster_p"          # Name of the new column for the cluster probabilities # nolint: line_length_linter.
  ) %>%
  mutate(cluster_number = factor(as.numeric(str_extract(cluster_number, "\\d+")))) %>%
  mutate(n_cluster = round(cluster_p*n)) %>%
  mutate(speakerid = as.factor(speakerid))
  word_sense_indexed <- word_long %>% filter(cluster_number == cluster)
  # Find optimal a-value
  ## Use a smaller subset of data for optimisation:
  filtered <- filter_down(word_sense_indexed, threshold)
  return(filtered)
}

get_high_change_senses <- function(word, window_size = 10, threshold = 0.3) {
  # Read the CSV file
  word_df <- read.csv(glue("./clusters/{word}_cluster_counts.csv"))
  
  # Identify the cluster probability columns
  cluster_p_columns <- grep("^cluster_.*_p$", colnames(word_df), value = TRUE)
  high_change_columns <- c()
  
  # Iterate over the identified columns
  for (column in cluster_p_columns) {
    years <- unique(word_df$year)
    mean_props <- c()
    
    # Calculate mean proportions over sliding windows
    for (start_year in years) {
      end_year <- start_year + window_size
      if (end_year <= max(years)) {
        window_data <- subset(word_df, year >= start_year & year < end_year)
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
  
  # Extract the cluster numbers from the column names
  cluster_numbers <- regmatches(high_change_columns, regexpr("\\d+", high_change_columns))
  
  return(cluster_numbers)
}

print(glue("Getting senses of '{word}' that significantly changed..."))
high_change_senses <- get_high_change_senses(word, window_size=window_size, threshold=change_threshold)
results_list = list()
json_filename = glue("{output_directory}/a-optim-{word}.json")
if (length(high_change_senses) < 1){
    print("None of the senses for this word showed enough change!")
} else{
    print(glue("Senses deemed to have shown enough change, using window size {window_size} and change threshold {change_threshold}:"))
    print(high_change_senses)
    if (file.exists(json_filename)){
      json_data <- fromJSON(json_filename)
      prior_data <- as_tibble(json_data)
    }
    for (i in 1:length(high_change_senses)){
        sense <- high_change_senses[i]
        print(glue("Starting on sense {sense}..."))
        try({
            if (exists('prior_data')){
              if (sense %in% prior_data$sense){
                print("Sense already finished on a previous run!")
                results_list[[i]] <- list(word = prior_data$word[[i]], sense = prior_data$sense[[i]], a_estimate = prior_data$a_estimate[[i]], ci_lower = prior_data$ci_lower[[i]], ci_upper = prior_data$ci_upper[[i]])
                next
              }
            }
            filtered <- get_data(word=word, cluster=sense, threshold=speaker_threshold)
            negative_log_likelihood <- function(a){
                compute_fREML(a, data = filtered)
            }
            lower_bound = -0.5
            upper_bound = 1.5
            starting_point = 0
            t1 <- Sys.time()
            mle_result <- mle(minuslogl = negative_log_likelihood, start = list(a = starting_point), method = "Brent",
                            lower = lower_bound, upper = upper_bound)
            t2 <- Sys.time()
            mle_summary <- summary(mle_result)
            a_estimate <- mle_summary@coef[1]
            std_err <- mle_summary@coef[2]
            ci_lower <- a_estimate-(1.96*std_err)
            ci_upper <- a_estimate+(1.96*std_err)
            print(glue("With threshold of {speaker_threshold}, optimization took this long:"))
            print(t2-t1)
            print(glue("Estimate of a: {a_estimate}"))
            print(glue("Standard error-based confidence intervals: {ci_lower}, {ci_upper}"))
            print("Writing to json file...")
            results_list[[i]] <- list(word = word, sense = sense, a_estimate = a_estimate, ci_lower = ci_lower, ci_upper = ci_upper)
            json_data <- toJSON(results_list, pretty = TRUE)
            write(json_data, file = glue("{output_directory}/a-optim-{word}.json"))
            print("Done!")
        })
    }
}

print("All done!")
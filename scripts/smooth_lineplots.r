library(glue)
library(mgcv)
library(tidyverse)
library(lme4)
library(stringr)
library(mgcViz)
library(jsonlite)

word_filepath <- "./assets/wordlist.txt"
words <- str_trim(readLines(word_filepath))

full_smoothplot_directory <- './plots/time-sense_line_smooths_full'
if (!file.exists(full_smoothplot_directory)) {
    dir.create(full_smoothplot_directory, recursive = TRUE)
    }

filtered_smoothplot_directory <- './plots/time-sense_line_smooths_filtered'
if (!file.exists(filtered_smoothplot_directory)) {
    dir.create(filtered_smoothplot_directory, recursive = TRUE)
    }


interpretable_plot_directory <- './plots/time-sense_line_smooths_simplified'
if (!file.exists(interpretable_plot_directory)) {
    dir.create(interpretable_plot_directory, recursive = TRUE)
    }

interpretable_word_sense_examples_path = './assets/interpretable_word_sense_examples.json'
interpretable_word_sense_examples <- fromJSON(interpretable_word_sense_examples_path)


get_high_change_senses <- function(word, window_size = 20, threshold = 0.3) {
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


for (word in words){
    word_tibble <- read.csv(glue("./clusters/{word}_cluster_counts.csv"))
    
    word_long <- word_tibble %>%
    select(-matches("^cluster_\\d+$")) %>%
    pivot_longer(
        cols = starts_with("cluster_"),  # Select columns that start with "cluster_"
        names_to = "cluster_number",     # Name of the new column for the cluster numbers # nolint: line_length_linter.
        names_prefix = "cluster_",       # Remove the prefix "cluster_" from the original column names
        values_to = "cluster_p"          # Name of the new column for the cluster probabilities # nolint: line_length_linter.
    ) %>%
    mutate(cluster_number = factor(as.numeric(str_extract(cluster_number, "\\d+")))) %>%
    mutate(n_cluster = round(cluster_p*n)) %>%
    mutate(age = as.numeric(age)) %>% # Also adding buckets for age group:
    mutate(age_group = cut(age, breaks = seq(20, 100, 20), include.lowest = TRUE)) %>%
    mutate(yob = as.numeric(yob)) %>%
    mutate(yob_group = cut(yob, breaks = seq(1800, 2000, 25), include.lowest = TRUE))
    
    # Basic Smooth Plots:
    basic_smoothplot <- word_long %>% ggplot(aes(x = year,y = cluster_p)) + geom_smooth(aes(color = cluster_number)) + ylim(0,1) + ylab("Proportion of Model-Predicted\nReplacement Words") + xlab("Year of Speech") + labs(color="Cluster Number")
    ggsave(glue("{full_smoothplot_directory}/{word}.png"), basic_smoothplot, width = 8, height = 5, dpi = 300)
    
    # Interpretable Sense Plots (if applicable) #TODO:
    if (word %in% names(interpretable_word_sense_examples)) {
        senses <- interpretable_word_sense_examples[[word]]
        word_long_interpretable <- word_long %>%
            filter(cluster_number %in% names(senses)) %>%
            mutate(cluster_number = as.character(cluster_number)) %>%
            mutate(cluster_number = as.character(senses[cluster_number]))
    #
    interpretable_smoothplot <- word_long_interpretable %>% ggplot(aes(x = year,y = cluster_p)) + geom_smooth(aes(color = cluster_number)) + ylim(0,1) + ylab("Probability of Word Sense") + xlab("Year of Speech") + labs(color="Word Sense", title=word)  + theme(text = element_text(size=10))
    ggsave(glue("{interpretable_plot_directory}/{word}.png"), interpretable_smoothplot, width = 8, height = 2.5, dpi = 300)
    }
    
    # Filtered Smooth Plots (only high change senses):
    high_change_senses <- get_high_change_senses(word)
    word_filtered <- word_long %>% filter(cluster_number %in% high_change_senses)
    if (nrow(word_filtered) > 0){
        filtered_smoothplot <- word_filtered %>% ggplot(aes(x = year,y = cluster_p)) + geom_smooth(aes(color = cluster_number)) + ylim(0,1) + ylab("Proportion of Model-Predicted\nReplacement Words") + xlab("Year of Speech") + labs(color="Cluster Number")
        ggsave(glue("{filtered_smoothplot_directory}/{word}.png"), filtered_smoothplot, width = 8, height = 3, dpi = 300)
    }
}

library(glue)
library(mgcv)
library(tidyverse)
library(lme4)
library(stringr)
library(mgcViz)
library(jsonlite)

word_filepath <- "./assets/wordlist.txt"
words <- str_trim(readLines(word_filepath))

smoothplot_directory <- './plots/time-sense_line_smooths'
if (!file.exists(smoothplot_directory)) {
    dir.create(smoothplot_directory, recursive = TRUE)
    }

interpretable_plot_directory <- './plots/time-sense_line_smooths_simplified'
if (!file.exists(interpretable_plot_directory)) {
    dir.create(interpretable_plot_directory, recursive = TRUE)
    }

interpretable_word_sense_examples_path = './assets/interpretable_word_sense_examples.json'
interpretable_word_sense_examples <- fromJSON(interpretable_word_sense_examples_path)

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
    ggsave(glue("{smoothplot_directory}/{word}.png"), basic_smoothplot, width = 8, height = 5, dpi = 300)
    
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
}

### This is a quick post-facto script to append to the collated_results.csv file some information about which year contained the highest predicted word sense probability.
library(tidyverse)
library(mgcv)
library(glue)
library(fs)
library(jsonlite)
library(stringr)

scratch_dir <- Sys.getenv('SCRATCH')
model_dir <- glue("{scratch_dir}/GAMMs")
word_filepath <- "./assets/wordlist.txt"
collated_results_path <- "collated_results.csv"

words <- str_trim(readLines(word_filepath))
collated_results <- read_csv(collated_results_path)

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


get_optimized_a_coef <- function(word, target_sense, a_optimization_results){
  word_results <- as_tibble(a_optimization_results[[word]])
  a_estimate <- word_results %>% filter(sense == target_sense) %>% .$a_estimate %>% .[[1]]
  return(a_estimate)
}

wordlist_for_processing = c()
senselist_for_processing = c()
peaklist_for_processing = c()

for (word in words){
    print(glue("Starting on {word}..."))
    word_tibble <- read.csv(glue("./clusters/{word}_cluster_counts.csv"))
    word_long <- word_tibble %>%
    select(-matches("^cluster_\\d+$")) %>% # Super roundabout, but we exclude the `cluster_<number>` columns here to recalculate them after pivoting
    pivot_longer(
        cols = starts_with("cluster_"),  # Select columns that start with "cluster_"
        names_to = "cluster_number",     # Name of the new column for the cluster numbers # nolint: line_length_linter.
        names_prefix = "cluster_",       # Remove the prefix "cluster_" from the original column names
        values_to = "cluster_p"          # Name of the new column for the cluster probabilities # nolint: line_length_linter.
    ) %>%
    mutate(cluster_number = factor(as.numeric(str_extract(cluster_number, "\\d+")))) %>%
    mutate(n_cluster = round(cluster_p*n)) %>%
    mutate(speakerid = as.factor(speakerid))
    print("Word data processed!")
    ### Pulling the cluster numbers that have been modelled:
    model_files <- dir_ls(model_dir)
    word_files <- model_files[grepl(glue("{word}-"), model_files) & grepl(glue("-LO_GAMM.rds"), model_files)]
    for (file in word_files){
        try({
            cluster <- str_extract(file, "(?<=-)[^-]+(?=-)")
            word_sense_indexed <- word_long %>% filter(cluster_number==cluster)
            #
            model <- readRDS(file)
            print("GAMM loaded successfully!")
            model_summary <- capture.output(summary(model, re.test=FALSE))
            model_summary_str <- paste(model_summary, collapse = "\n")
            #
            yob_range <- seq(min(word_sense_indexed$yob), max(word_sense_indexed$yob), by=1)
            year_range <- seq(min(word_sense_indexed$year), max(word_sense_indexed$year), by=1)
            age_min <- round(mean(word_sense_indexed$age) - 3*sd(word_sense_indexed$age))
            age_max <- round(mean(word_sense_indexed$age) + 3*sd(word_sense_indexed$age))
            #
            a_value <- get_optimized_a_coef(word, cluster, a_optimization_results)
            pred_data <- expand.grid(yob = yob_range, year = year_range) %>%
            mutate(age = year - yob) %>% 
            mutate(speakerid = first(word_sense_indexed$speakerid)) %>%# Just so the mgcv:predict fn doesn't throw up an error asking for speakerid values -- this won't matter because we exclude the by-speaker effects ## Confirmed later, changing this doesn't affect preds
            filter(age >= age_min & age <= age_max) %>%
            mutate(time = year - (a_value * age)) #%>%
            #select(c("time", "speakerid"))
            #
            preds <- mgcv::predict.bam(model, newdata=pred_data, type="response", exclude='s(speakerid)')
            pred_data$probability = preds
            print("Prediction data generated!")
            #
            peak_year = pred_data %>%
                group_by(year) %>%
                summarise(mean_p = mean(probability)) %>%
                filter(mean_p == max(mean_p)) %>%
                .$year
            
            wordlist_for_processing <- c(wordlist_for_processing, word)
            senselist_for_processing <- c(senselist_for_processing, cluster)
            peaklist_for_processing <- c(peaklist_for_processing, peak_year)
            print(glue("Done with sense {cluster}"))
            rm(model, word_sense_indexed)
            gc()
        })
    } 
    rm(word_long, word_tibble)
    gc()
}


prediction_year_peaks <- data.frame(word = wordlist_for_processing,
                                    sense = senselist_for_processing,
                                    peak_year = peaklist_for_processing) %>%
                                    mutate(sense = as.numeric(sense))

prediction_year_peaks %>% write_csv("prediction_peaks.csv")

collated_results_peaks_added <- collated_results %>% 
    filter(word %in% words) %>% 
    left_join(prediction_year_peaks, by=c("word", "sense")) %>% 
    select(word, sense, a_estimate, peak_year)

collated_results_peaks_added %>% write_csv("collated_results_peaks_added.csv")
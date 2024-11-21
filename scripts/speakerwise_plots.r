library(glue)
library(tidyverse)
library(jsonlite)


speakerwise_plot_directory = "./plots/speakerwise_plots"
if (!file.exists(speakerwise_plot_directory)) {
    dir.create(speakerwise_plot_directory, recursive = TRUE)
    }

speakerwise_examples_path <- "./assets/speakerwise_word_sense_examples.json"
speakerwise_examples_json_data <- fromJSON(speakerwise_examples_path)

df <- as.data.frame(do.call(rbind, speakerwise_examples_json_data)) %>%
  mutate(word = rownames(.)) %>%
  select(word, start_year, end_year, sense, interpretable_sense)

rownames(df) <- NULL

extra_speaker_bioinfo_path = "./assets/additional_speaker_bioinformation.json"
bioinfo_json_data <- fromJSON(extra_speaker_bioinfo_path)
bioinfo_df <- as.data.frame(do.call(rbind, bioinfo_json_data))
bioinfo_df$birth <- as.character(bioinfo_df$birth)
bioinfo_df$death <- as.character(bioinfo_df$death)

for (row in 1:nrow(df)){
    word <- as.character(df[row, "word"])
    cluster <- as.character(df[row, "sense"])
    interpretable_sense <- as.character(df[row, "interpretable_sense"])
    start_year <- as.numeric(df[row, "start_year"])
    end_year <- as.numeric(df[row, "end_year"])

    word_tibble <- read_csv(glue("./clusters/{word}_cluster_counts.csv"))

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
        mutate(age_group = cut(age, breaks = seq(25, 85, 20), include.lowest = TRUE)) %>%
        mutate(yob = as.numeric(yob)) %>%
        mutate(yob_group = cut(yob, breaks = seq(1800, 2000, 25), include.lowest = TRUE))

    top_speakers <- word_long %>% 
        group_by(speakerid) %>%
        summarise(n_total = sum(n), firstname = first(firstname), lastname = first(lastname), yob = first(yob), first_use = min(year), last_use = max(year)) %>%
        arrange(desc(n_total)) %>% 
        filter(first_use > start_year-10, first_use < end_year-10) %>%
        head(6)
    title_string = glue("Speaker-wise probability of {interpretable_sense} sense of {word} being used:")
    
    population_smooth_data <- word_long %>% filter(cluster_number == cluster)
    
    plot <- word_long %>%
        filter(speakerid %in% top_speakers$speakerid) %>%
        filter(cluster_number == cluster) %>%
        mutate(full_name=paste(firstname, lastname)) %>%
        mutate(
            common_name = map_chr(full_name, ~ bioinfo_df[[.x, "name"]]), 
            birth_year = map_chr(full_name, ~ bioinfo_df[[.x, "birth"]]), 
            death_year = map_chr(full_name, ~ bioinfo_df[[.x, "death"]])
            ) %>%
        mutate(name_string = glue("{common_name} ({birth_year}-{death_year})")) %>%
        ggplot(aes(x=year, y=cluster_p, colour=name_string)) +
            geom_smooth() +
            geom_smooth(data = population_smooth_data, aes(x = year, y = cluster_p), 
                    color = "black", linetype = "dotted", inherit.aes = FALSE) +
            xlim(min(top_speakers$first_use)-10,min(2010, max(top_speakers$last_use)+10)) +
            ylim(0,1) +
            labs(
                title = title_string,
                x = "Year of Speech",
                y = "Probability of Word Sense Given Use",
                color = "Speaker"
                )
    filepath <- glue("{speakerwise_plot_directory}/speakerwise-{word}-{cluster}.png")
    ggsave(filepath, plot, width=8, height=6, dpi=300)

}

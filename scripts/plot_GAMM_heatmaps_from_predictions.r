library(tidyverse)
library(viridis)
library(glue)

prediction_datasets_dir <- Sys.getenv('PREDICTION_DATASETS_DIR')
plots_dir <- Sys.getenv('PLOTS_DIR')
model_type <- Sys.getenv('MODEL_TYPE')

# Loop through each prediction dataset:
prediction_datasets <- list.files(prediction_datasets_dir, pattern = "\\.csv$", full.names = TRUE)

for (prediction_dataset in prediction_datasets) {
    # Read the dataset
    pred_data <- read.csv(prediction_dataset)
    # Extract the word and cluster from the filename: filenames are of format <word>-<cluster>.csv
    filename <- basename(prediction_dataset) 
    word <- str_extract(filename, "^[^-]+")
    cluster <- str_extract(filename, "(?<=-)[^-]+(?=\\.csv)")
    plot_filename <- glue("{plots_dir}/{word}-{cluster}-{model_type}_plot.pdf")
    print(glue("Processing {word} - {cluster}..."))
    plot <- ggplot(pred_data, aes(x = age, y = year, fill = probability)) + 
        geom_tile() +
        scale_fill_gradientn(
            name = "Predicted\nProbability",
            colours = c("#3E5088", "#2A8E89", "#6BC65E", "#F8E527"),
            limits = c(0, 1)
        ) +
        geom_contour(aes(z = probability), color = 'black') +
        labs(x = "Age", y = "Year", title = glue("Model-Predicted Probability of\nSense {cluster} of {word}")) +
        theme_minimal() +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 18),
            axis.text.y = element_text(size = 18),
            axis.title.x = element_text(size = 18),
            axis.title.y = element_text(size = 18)
        )
    print("Plot created!")
    print("Saving plot...")
    ggsave(plot_filename, plot, width = 8, height = 6, dpi = 900)
    print(glue("Plot saved as {plot_filename}"))
}


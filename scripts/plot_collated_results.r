library(tidyverse)
library(glue)

collated_results_plot_directory <- "./plots/collated_results_plots"
if (!dir.exists(collated_results_plot_directory)){
    dir.create(collated_results_plot_directory)}

collated_results <- read_csv("./collated_results.csv")

collated_results <- collated_results %>%
    mutate(log_overall_word_freq = log(overall_word_freq), log_cluster_substitution_total = log(cluster_substitution_total))

# a-estimates as function of total substitutions from the cluster (i.e. the raw prevalence of the cluster)
plot <- ggplot(collated_results, aes(x=cluster_substitution_total, y=a_estimate, colour=sig_non_zero)) +
    geom_point(size=1) + 
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=20), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=16), #change font size of legend text
        legend.title=element_text(size=16),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="Total Substitutions From Cluster",
    y="a-Estimate",
    colour="a-Estimate\nSignificantly\nNon-Zero?")

ggsave(glue("{collated_results_plot_directory}/a_estim_sense_freq_scatter.png"), plot, width=12, height=6, dpi=300)

# a-estimates as function of total substitutions from the cluster (i.e. the raw prevalence of the cluster) (LOG)
plot <- ggplot(collated_results, aes(x=log_cluster_substitution_total, y=a_estimate, colour=sig_non_zero)) +
    geom_point(size=1) + 
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=20), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=16), #change font size of legend text
        legend.title=element_text(size=16),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="log Total Substitutions From Cluster",
    y="a-Estimate",
    colour="a-Estimate\nSignificantly\nNon-Zero?")

ggsave(glue("{collated_results_plot_directory}/a_estim_log_sense_freq_scatter.png"), plot, width=12, height=6, dpi=300)

# a-estimates as function of total number of uses of word
plot <- ggplot(collated_results, aes(x=overall_word_freq, y=a_estimate, colour=sig_non_zero)) +
    geom_point(size=1) + 
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=20), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=16), #change font size of legend text
        legend.title=element_text(size=16),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="Total Uses of Word",
    y="a-Estimate",
    colour="a-Estimate\nSignificantly\nNon-Zero?")

ggsave(glue("{collated_results_plot_directory}/a_estim_word_freq_scatter.png"), plot, width=12, height=6, dpi=300)


# a-estimates as function of total number of uses of word (LOG)
plot <- ggplot(collated_results, aes(x=log_overall_word_freq, y=a_estimate, colour=sig_non_zero)) +
    geom_point(size=1) + 
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=20), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=16), #change font size of legend text
        legend.title=element_text(size=16),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="log Total Uses of Word",
    y="a-Estimate",
    colour="a-Estimate\nSignificantly\nNon-Zero?")

ggsave(glue("{collated_results_plot_directory}/a_estim_log_word_freq.png"), plot, width=12, height=6, dpi=300)


# C.I. width as function of overall word frequency (log)
plot <- ggplot(collated_results, aes(x=ci_width, y=log_overall_word_freq)) +
    geom_point(size=0.75) +
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=20), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=20), #change font size of legend text
        legend.title=element_text(size=20),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="C.I. Width",
    y="log of Overall Word Freq") 

ggsave(glue("{collated_results_plot_directory}/ci_width_word_freq_scatter.png"), plot, width=12, height=6, dpi=300)

# C.I. width as function of estimated a-value
plot <- ggplot(collated_results, aes(x=ci_width, y=a_estimate, colour=sig_non_zero)) +
    geom_point(size=0.75) +
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=16), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=12), #change font size of legend text
        legend.title=element_text(size=12),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="C.I. Width",
    y="a-Estimate",
    colour="a-Estimate\nSignificantly\nNon-Zero?") 

ggsave(glue("{collated_results_plot_directory}/ci_width_a_estimate.png"), plot, width=12, height=8, dpi=300)

# LO and 2D GAMM Adjusted R-Sq Density Plots:
## 2D GAMM Adjusted R-Sq:
plot <- ggplot(collated_results, aes(x=r_sq_2D)) +
    geom_density() +
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=18), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=16), #change font size of legend text
        legend.title=element_text(size=16),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="Adjusted R-Sq. from 2D GAMM")

ggsave(glue("{collated_results_plot_directory}/2D_r_sq_density.png"), plot, width=8, height=6, dpi=300)

## LO GAMM Adjusted R-Sq:
plot <- ggplot(collated_results, aes(x=r_sq_LO)) +
    geom_density() +
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=18), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=16), #change font size of legend text
        legend.title=element_text(size=16),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="Adjusted R-Sq. from 2D GAMM")

ggsave(glue("{collated_results_plot_directory}/LO_r_sq_density.png"), plot, width=8, height=6, dpi=300)


# Density plot of Normalized L2 Distance b/w 2D and LO GAMM:
plot <- ggplot(collated_results, aes(x=l2_distance_normalized)) +
    geom_density() +
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=18), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=16), #change font size of legend text
        legend.title=element_text(size=16),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="Normalized L2 Distance b/w 2D and LO GAMM Predictions")

ggsave(glue("{collated_results_plot_directory}/normalized_l2_density.png"), plot, width=8, height=6, dpi=300)



# Correlations between LO and 2D GAMM fit metrics:
## Adj. R.Sq:
plot <- ggplot(collated_results, aes(x=r_sq_LO, y=r_sq_2D, colour=sig_non_zero)) +
    geom_point() + 
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=18), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=16), #change font size of legend text
        legend.title=element_text(size=16),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="Adjusted R-Sq. from LO GAMM",
    y="Adjusted R-Sq. from 2D GAMM",
    colour="a-Estimate\nSignificantly\nNon-Zero?")

ggsave(glue("{collated_results_plot_directory}/LO_vs_2D_r_sq_scatter.png"), plot, width=8, height=6, dpi=300)

## fREML:
plot <- ggplot(collated_results, aes(x=freml_LO, y=freml_2D, colour=sig_non_zero)) +
    geom_point() + 
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=18), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=16), #change font size of legend text
        legend.title=element_text(size=16),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="fREML from LO GAMM",
    y="fREML from 2D GAMM",
    colour="a-Estimate\nSignificantly\nNon-Zero?")

ggsave(glue("{collated_results_plot_directory}/LO_vs_2D_freml_scatter.png"), plot, width=8, height=6, dpi=300)

## Deviance Explained
plot <- ggplot(collated_results, aes(x=dev_exp_LO, y=dev_exp_2D, colour=sig_non_zero)) +
    geom_point() + 
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=18), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=16), #change font size of legend text
        legend.title=element_text(size=16),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="Deviance Explained by LO GAMM",
    y="Deviance Explained by 2D GAMM",
    colour="a-Estimate\nSignificantly\nNon-Zero?") 

ggsave(glue("{collated_results_plot_directory}/LO_vs_2D_dev_exp_scatter.png"), plot, width=8, height=6, dpi=300)

# Histogram of a-estimates:
plot <- ggplot(collated_results, aes(x=a_estimate, fill=sig_non_zero)) + 
    geom_histogram(breaks=seq(-0.4, 1.6, 0.1)) + 
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=20), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=16), #change font size of legend text
        legend.title=element_text(size=16),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="a-Estimate",
    fill="a-Estimate\nSignificantly\nNon-Zero?") 

ggsave(glue("{collated_results_plot_directory}/a_estimate_hist.png"), plot, width=8, height=6, dpi=300)

# Density plot of a-estimates:
plot <- ggplot(collated_results, aes(x=a_estimate)) + 
    geom_density() + 
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=20), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=16), #change font size of legend text
        legend.title=element_text(size=16),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="Linear Offset Estimate") 

ggsave(glue("{collated_results_plot_directory}/a_estimate_density.png"), plot, width=8, height=6, dpi=300)

plot <- ggplot(collated_results, aes(x = a_estimate, colour = sig_non_zero)) + 
    geom_density(aes(y = after_stat(count)), position = "identity") + 
    theme(text = element_text(size = 20), 
          axis.text = element_text(size = 20), 
          axis.title = element_text(size = 20), 
          plot.title = element_text(size = 20), 
          legend.text = element_text(size = 16), 
          legend.title = element_text(size = 16), 
          plot.margin = margin(1, 1, 1, 1, "cm")) + 
          labs(x = "Linear Offset Estimate",
            y = "Count Density",
            colour = "Estimate\nSignificantly\nNon-Zero?")

ggsave(glue("{collated_results_plot_directory}/a_estimate_density_scaled.png"), plot, width=8, height=6, dpi=300)


# Alpha-Values against R-Sq or Normalized L2 Distances:
## Against Normalized L2 Distance:
plot <- ggplot(collated_results, aes(x=l2_distance_normalized, y=a_estimate)) +
    geom_point(size=0.75) +
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=16), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=12), #change font size of legend text
        legend.title=element_text(size=12),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="Normalized L2 Distance b/w 2D and LO GAMM Predictions",
    y="a-Estimate") 

ggsave(glue("{collated_results_plot_directory}/l2_distance_a_estimate.png"), plot, width=12, height=8, dpi=300)


## Against Adjusted R-Sq:
plot <- ggplot(collated_results, aes(x=r_sq_LO, y=a_estimate)) +
    geom_point(size=0.75) +
    theme(text=element_text(size=20), #change font size of all text
        axis.text=element_text(size=20), #change font size of axis text
        axis.title=element_text(size=16), #change font size of axis titles
        plot.title=element_text(size=20), #change font size of plot title
        legend.text=element_text(size=12), #change font size of legend text
        legend.title=element_text(size=12),
        plot.margin=margin(1,1,1,1, "cm")) + #change font size of legend title 
    labs(x="Adjusted R-Squared",
    y="a-Estimate") 

ggsave(glue("{collated_results_plot_directory}/r_sq_a_estimate.png"), plot, width=12, height=8, dpi=300)


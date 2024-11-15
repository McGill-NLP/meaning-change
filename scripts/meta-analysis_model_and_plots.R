library(tidyverse)
library(brms)

# Collating results and adding peak probability years:
collated <- read_csv("collated_results.csv")
collated <- collated %>% mutate(std_err = (a_estimate-ci_lower)/1.96,
                                sq_error = (mean(a_estimate) - a_estimate)^2, 
                                log_word_freq = log(overall_word_freq),
                                word = as.factor(word),
                                word_sense = as.factor(paste(word, as.character(sense))))

# prediction_peaks <- read_csv("prediction_peaks.csv")

# collated <- collated %>% left_join(prediction_peaks, by=c("word", "sense"))


# Fitting Bayesian meta-analysis model:
priors <- c(prior(normal(0, 1), class = Intercept),
                prior(normal(0, 1), class = sd))

fit_ma <- brm(a_estimate | resp_se(std_err, sigma = FALSE) ~
                 1 + (1 | word_sense),
               data = collated,
               prior = priors,
               control = list(adapt_delta = .99, max_treedepth = 20),
               iter = 3000,
               file = "meta-analysis.rds")

############### Details of meta-analysis model ###################
summary(fit_ma)
##################################################################


# Miscellaneous plots:
df_ma <- collated %>%
  mutate(Q2.5 = a_estimate - 1.96 * std_err,
         Q97.5 = a_estimate + 1.96 * std_err,
         Estimate = a_estimate,
         type = "original") %>% 
  drop_na(std_err)

df_Intercept <- posterior_summary(fit_ma,
                                      variable = c("b_Intercept")) %>%
  as.data.frame() %>%
  mutate(publication = "M.A. estimate", type = "")

df_model <- fitted(fit_ma) %>%
  # Convert matrix to data frame:
  as.data.frame() %>%
  # Add a column to identify the estimates,
  # and another column to identify the publication:
  mutate(type = "adjusted",
         word_sense = df_ma$word_sense)


df_plotting <- df_ma %>% mutate(adjusted_a_estimate = df_model$Estimate,
                 adjusted_ci_lower = df_model$Q2.5,
                 adjusted_ci_upper = df_model$Q97.5)

# Ordering by peak year
# ggplot(df_plotting, aes(x=peak_year, y=adjusted_a_estimate)) +
#  geom_point() + 
#  geom_segment(aes(x=peak_year, xend=peak_year, y=0, yend=adjusted_a_estimate))

# Ordering by overall word freq
ggplot(df_plotting, aes(x=overall_word_freq, y=adjusted_a_estimate)) +
  geom_point() + 
  geom_segment(aes(x=overall_word_freq, xend=overall_word_freq, y=0, yend=adjusted_a_estimate))

# Ordering by log word freq
ggplot(df_plotting, aes(x=log_word_freq, y=adjusted_a_estimate)) +
  geom_point() + 
  geom_segment(aes(x=log_word_freq, xend=log_word_freq, y=0, yend=adjusted_a_estimate))


# Ascending order
plot <- df_plotting %>% 
  arrange(a_estimate) %>% 
  mutate(order = row_number()) %>% 
  ggplot(aes(x=order, y=a_estimate)) +
    geom_point() + 
    geom_segment(aes(x=order, xend=order, y=0, yend=a_estimate)) +  # Lollipops 
    # geom_segment(aes(x=0, xend=max(order), y=df_Intercept$Estimate)) + 
    ggtitle("")

plot_title <- "./plots/collated_results_plots/lollipop.png"
ggsave(plot_title, plot, width=8, height=6, dpi=300)

# Ascending order with more labels:
df_plotting %>% 
  arrange(a_estimate) %>% 
  mutate(order = row_number()) %>% 
  ggplot(aes(x = order, y = a_estimate)) +
  geom_point() + 
  labs(x="", y="") +
  geom_segment(aes(x = order, xend = order, y = 0, yend = a_estimate)) +  # Lollipops
  geom_segment(aes(x = 0, xend = max(order), y = df_Intercept$Estimate, 
                   yend = df_Intercept$Estimate), color = "red", size = 1, linetype = "solid") +  # Red line for estimate
  geom_ribbon(aes(ymin = df_Intercept$Q2.5, ymax = df_Intercept$Q97.5, 
                  fill = "Confidence Interval"), 
              xmin = 0, xmax = max(df_plotting$order),
              alpha = 0.2) +  # Translucent band for C.I.
  #scale_fill_manual(name = "Legend", values = c("Confidence Interval" = "red")) +  # Legend for CI
  #scale_color_manual(name = "Legend", values = c("Estimate" = "red")) +  # Legend for Estimate line
  theme(axis.text.x = element_blank(),  # Remove x-axis text
        axis.ticks.x = element_blank(),
        axis.title.y = element_text(size=18),
        axis.text.y = element_text(size=24)) #+  # Remove x-axis ticks
  #guides(fill = guide_legend(title = "Estimated Mean and C.I."), 
         #color = guide_legend(title = "Estimated Mean and C.I."))



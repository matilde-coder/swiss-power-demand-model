# Interpretable regression: how strongly does temperature drive Swiss
# electricity demand, has that changed over time, and what is the
# weather-adjusted demand level per year?
#
# Model (OLS, all years 2020-2025, explanatory rather than forecasting):
#   demand_gwh ~ HDD x year + CDD + sunshine + weekday + month + holidays + year
# HDD = heating degree days (20 - T when T < 12 C), CDD = cooling degree days.
#
# Outputs: results/r_coefficients.csv, results/r_hdd_by_year.csv,
#          results/r_summary.md, figures/05-07.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

root <- tryCatch(
  normalizePath(file.path(dirname(sys.frame(1)$ofile), "..")),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    f <- sub("^--file=", "", args[grep("^--file=", args)])
    if (length(f)) normalizePath(file.path(dirname(f), "..")) else getwd()
  }
)
res_dir <- file.path(root, "results")
fig_dir <- file.path(root, "figures")
dir.create(res_dir, showWarnings = FALSE)
dir.create(fig_dir, showWarnings = FALSE)

ink <- "#0b0b0b"; muted <- "#52514e"; grid_col <- "#e4e3df"; blue <- "#2a78d6"
theme_report <- theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold", colour = ink, size = 11),
    plot.title.position = "plot",
    plot.subtitle = element_text(colour = muted),
    axis.text = element_text(colour = muted),
    axis.title = element_text(colour = muted),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(colour = grid_col, linewidth = 0.4),
    plot.background = element_rect(fill = "white", colour = NA)
  )

weekdays_en <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
d <- read_csv(file.path(root, "data", "daily.csv"), show_col_types = FALSE) |>
  mutate(
    weekday = factor(weekday, levels = weekdays_en),
    month = factor(month),
    year_f = factor(year)
  )

# ---- 1. Main model -----------------------------------------------------------
fit <- lm(demand_gwh ~ hdd * year_f + cdd + sunshine_h + weekday + month +
            is_holiday + is_xmas_break, data = d)

ci <- confint(fit)
coefs <- tibble(
  term = names(coef(fit)),
  estimate = unname(coef(fit)),
  ci_low = ci[, 1],
  ci_high = ci[, 2],
  p_value = summary(fit)$coefficients[, 4]
) |>
  mutate(across(where(is.numeric), \(x) signif(x, 4)))
write_csv(coefs, file.path(res_dir, "r_coefficients.csv"))

# Heating sensitivity per year = base HDD slope + year interaction.
# Variances from the coefficient covariance matrix give a proper CI.
V <- vcov(fit)
years <- levels(d$year_f)
hdd_by_year <- bind_rows(lapply(years, function(y) {
  w <- setNames(rep(0, length(coef(fit))), names(coef(fit)))
  w["hdd"] <- 1
  inter <- paste0("hdd:year_f", y)
  if (inter %in% names(w)) w[inter] <- 1
  est <- sum(w * coef(fit))
  se <- sqrt(drop(t(w) %*% V %*% w))
  tibble(year = as.integer(y), gwh_per_hdd = est,
         ci_low = est - 1.96 * se, ci_high = est + 1.96 * se)
}))
write_csv(mutate(hdd_by_year, across(-year, \(x) round(x, 3))),
          file.path(res_dir, "r_hdd_by_year.csv"))

# Weather- and calendar-adjusted demand level per year: predict every year on
# the *same* reference calendar and weather (2020-2025 average day mix).
ref <- d
level_by_year <- bind_rows(lapply(years, function(y) {
  nd <- mutate(ref, year_f = factor(y, levels = years))
  tibble(year = as.integer(y), adj_gwh_per_day = mean(predict(fit, nd)))
})) |>
  mutate(index_2020 = 100 * adj_gwh_per_day / first(adj_gwh_per_day))

# ---- 2. Figures ----------------------------------------------------------------
# 05: temperature response on normal working days (Tue-Fri, no holidays).
wd <- d |> filter(weekday %in% c("Tuesday", "Wednesday", "Thursday", "Friday"),
                  is_holiday == 0, is_xmas_break == 0)
p5 <- ggplot(wd, aes(temp_c, demand_gwh)) +
  geom_point(colour = blue, alpha = 0.35, size = 1.1) +
  geom_smooth(method = "loess", se = FALSE, colour = ink, linewidth = 0.9, span = 0.5,
              formula = y ~ x) +
  geom_vline(xintercept = 12, colour = muted, linetype = "dashed", linewidth = 0.4) +
  annotate("text", x = 12.3, y = max(wd$demand_gwh), hjust = 0, size = 3,
           colour = muted, label = "heating threshold 12 °C") +
  labs(title = "Colder days, higher demand",
       subtitle = "Daily Swiss end-user electricity demand on working days (Tue-Fri), 2020-2025",
       x = "Population-weighted mean temperature (°C)", y = "GWh per day") +
  theme_report
ggsave(file.path(fig_dir, "05_temperature_response.png"), p5, width = 7, height = 4, dpi = 150)

p6 <- ggplot(hdd_by_year, aes(factor(year), gwh_per_hdd)) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.15, colour = muted) +
  geom_point(colour = blue, size = 3) +
  labs(title = "Heating sensitivity per year",
       subtitle = "Extra GWh per day for each heating degree day, with 95 % confidence interval",
       x = NULL, y = "GWh per heating degree day") +
  theme_report
ggsave(file.path(fig_dir, "06_heating_sensitivity_by_year.png"), p6, width = 6, height = 3.5, dpi = 150)

# Dots + line rather than bars: the interesting range is a few percent, and
# bars would need a zero baseline that hides it.
p7 <- ggplot(level_by_year, aes(year, index_2020)) +
  geom_hline(yintercept = 100, colour = muted, linewidth = 0.4) +
  geom_line(colour = blue, linewidth = 0.8) +
  geom_point(colour = blue, size = 3) +
  geom_text(aes(label = sprintf("%.1f", index_2020)), vjust = -1.1, size = 3.2, colour = ink) +
  scale_x_continuous(breaks = level_by_year$year) +
  scale_y_continuous(expand = expansion(mult = 0.25)) +
  labs(title = "Weather-adjusted demand level",
       subtitle = "Same weather and calendar every year; index 2020 = 100",
       x = NULL, y = "Index") +
  theme_report
ggsave(file.path(fig_dir, "07_weather_adjusted_level.png"), p7, width = 6, height = 3.5, dpi = 150)

# ---- 3. Plain-language summary ---------------------------------------------------
get <- function(t) coefs$estimate[coefs$term == t]
mean_winter <- mean(d$demand_gwh[d$season == "winter"])
hdd_last <- tail(hdd_by_year, 1)
lines <- c(
  "# Regression summary (R)",
  "",
  sprintf("Model: OLS on %d days, adjusted R² = %.3f, residual SE = %.2f GWh/day.",
          nrow(d), summary(fit)$adj.r.squared, summary(fit)$sigma),
  "",
  "## Key effects",
  "",
  sprintf("- **Temperature (%d):** each heating degree day (1 °C colder on a heating day) adds **%.2f GWh/day** (95 %% CI %.2f to %.2f), about %.1f %% of an average winter day.",
          hdd_last$year, hdd_last$gwh_per_hdd, hdd_last$ci_low, hdd_last$ci_high,
          100 * hdd_last$gwh_per_hdd / mean_winter),
  sprintf("- **Heating sensitivity over time:** %.2f GWh per HDD in %d vs %.2f in %d.",
          hdd_by_year$gwh_per_hdd[1], hdd_by_year$year[1],
          hdd_last$gwh_per_hdd, hdd_last$year),
  sprintf("- **Cooling:** each cooling degree day adds %.2f GWh/day.", get("cdd")),
  sprintf("- **Sunshine:** each extra hour of sunshine changes demand by %.2f GWh/day.", get("sunshine_h")),
  sprintf("- **Sunday vs Monday:** %.1f GWh/day; public holiday: %.1f GWh/day.",
          get("weekdaySunday"), get("is_holiday")),
  "",
  "## Weather-adjusted demand level (index 2020 = 100)",
  "",
  "| Year | GWh/day | Index |",
  "|---|---:|---:|",
  sprintf("| %d | %.1f | %.1f |", level_by_year$year, level_by_year$adj_gwh_per_day,
          level_by_year$index_2020)
)
writeLines(lines, file.path(res_dir, "r_summary.md"))
cat(lines, sep = "\n")

# --------------------------------------------------------------------------
# Project: TR 2025-26 Chapter 4
# Author: Pablo Garcia Guzman
# This script: Visualize MPD per-code availability across election years
# --------------------------------------------------------------------------

library(tidyverse)
library(ggtext)

source("/Users/pablogguz_/Documents/GitHub/poli-io-new/pablo/_aux/_theme_oce.r")

root <- "/Users/pablogguz_/Documents/GitHub/poli-io-new/pablo/"
fig_path <- paste0(root, "output/figures/")

# Load availability data
df <- read_csv(paste0(root, "output/tables/mpd_code_availability.csv"))

# Reshape to long
df_long <- df %>%
  pivot_longer(-election_year, names_to = "code", values_to = "share_available") %>%
  mutate(code = str_remove(code, "^avail_"))

# Define the index membership and labels
econ_codes <- tribble(
  ~code,      ~sign, ~label,                                          ~group,
  "per401",   "-",   "per401: Free market economy",                   "Market vs state",
  "per403",   "+",   "per403: Market regulation",                     "Market vs state",
  "per405",   "+",   "per405: Corporatism",                           "Market vs state",
  "per412",   "+",   "per412: Direct govt control",                   "Market vs state",
  "per413",   "+",   "per413: Nationalisation",                       "Market vs state",
  "per414",   "-",   "per414: Economic orthodoxy",                    "Market vs state",
  "per415",   "+",   "per415: Marxist-Leninist",                      "Market vs state",
  "per4012",  "-",   "per4012: Oppose govt control*",                 "Market vs state",
  "per4123",  "+",   "per4123: Public industries*",                   "Market vs state",
  "per4124",  "+",   "per4124: Socialist property*",                  "Market vs state",
  "per503",   "+",   "per503: Social justice",                        "Redistribution / welfare",
  "per504",   "+",   "per504: Welfare expansion",                     "Redistribution / welfare",
  "per505",   "-",   "per505: Welfare limitation",                    "Redistribution / welfare",
  "per506",   "+",   "per506: Education expansion",                   "Redistribution / welfare",
  "per507",   "-",   "per507: Education limitation",                  "Redistribution / welfare",
  "per701",   "+",   "per701: Pro-labour",                            "Redistribution / welfare",
  "per702",   "-",   "per702: Anti-labour",                           "Redistribution / welfare",
  "per5041",  "-",   "per5041: Private welfare*",                     "Redistribution / welfare",
  "per5061",  "-",   "per5061: Private education*",                   "Redistribution / welfare",
)

cult_codes <- tribble(
  ~code,      ~sign, ~label,                                          ~group,
  "per107",   "-",   "per107: Intl cooperation",                      "Globalisation / EU",
  "per108",   "-",   "per108: Pro-EU",                                "Globalisation / EU",
  "per109",   "+",   "per109: Sovereignty",                           "Globalisation / EU",
  "per110",   "+",   "per110: Anti-EU",                               "Globalisation / EU",
  "per601",   "+",   "per601: Nationalism",                           "Identity / citizenship",
  "per602",   "-",   "per602: Anti-nationalism",                      "Identity / citizenship",
  "per2022",  "+",   "per2022: Restrictive citizenship*",             "Identity / citizenship",
  "per2023",  "-",   "per2023: Lax citizenship*",                     "Identity / citizenship",
  "per603",   "+",   "per603: Traditional morality",                  "Morality / diversity",
  "per604",   "-",   "per604: Social liberalism",                     "Morality / diversity",
  "per607",   "-",   "per607: Multiculturalism",                      "Morality / diversity",
  "per608",   "+",   "per608: Assimilation",                          "Morality / diversity",
  "per7062",  "-",   "per7062: Refugee assistance*",                  "Morality / diversity",
)

# Function to create the heatmap
make_coverage_plot <- function(df_long, code_info, title_text) {

  df_plot <- df_long %>%
    inner_join(code_info, by = "code") %>%
    mutate(
      label = factor(label, levels = rev(code_info$label)),
      pct = share_available * 100
    )

  ggplot(df_plot, aes(x = election_year, y = label, fill = pct)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = ifelse(pct == 100, "", sprintf("%.0f", pct))),
              size = 2.7, color = "white") +
    scale_fill_gradient2(
      low = "#E41E25", mid = "#ffad41", high = "#00448D",
      midpoint = 50,
      limits = c(0, 100),
      name = "% of party-elections\nwith data"
    ) +
    scale_x_continuous(breaks = sort(unique(df_plot$election_year))) +
    labs(
      title = title_text,
      x = "Election year",
      y = NULL,
      caption = "* CEE sub-codes (only coded for Central & Eastern European elections, 0% availability from ~2015). Blank cells = 100% available."
    ) +
    theme_minimal(base_size = 16, base_family = "Libre Franklin") +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      legend.position = "right",
      legend.direction = "vertical",
      legend.key.height = unit(1.5, "cm"),
      panel.grid = element_blank(),
      plot.background = element_rect(fill = "white", color = NA),
      plot.title = element_text(size = 22, face = "bold", hjust = 0),
      plot.caption = element_text(size = 10, hjust = 0, face = "italic")
    )
}

# Create economic index coverage plot
p_econ <- make_coverage_plot(
  df_long, econ_codes,
  "Economic index: MPD per-code availability by election year"
)

ggsave(paste0(fig_path, "mpd_coverage_economic.png"), p_econ,
       width = 14, height = 7, dpi = 300, bg = "white")

# Create cultural index coverage plot
p_cult <- make_coverage_plot(
  df_long, cult_codes,
  "Cultural index: MPD per-code availability by election year"
)

ggsave(paste0(fig_path, "mpd_coverage_cultural.png"), p_cult,
       width = 14, height = 6, dpi = 300, bg = "white")

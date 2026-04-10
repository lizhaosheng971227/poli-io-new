#-----------------------------------------------------------------------------
# Project: ESS-MPD paper
# Author: Pablo Garcia Guzman
# This script: Chart of economic and cultural polarization over time (p80-p20)
#-----------------------------------------------------------------------------

packages_to_load <- c(
    "tidyverse",
    "scales",
    "data.table",
    "haven",
    "ggtext",
    "RColorBrewer",
    "ggrepel",
    "writexl"
)

package.check <- lapply(
    packages_to_load,
    FUN = function(x) {
        if (!require(x, character.only = TRUE)) {
            install.packages(x, dependencies = TRUE)
        }
    }
)

lapply(packages_to_load, require, character = T)

source("/Users/pablogguz_/Documents/GitHub/poli-io-new/pablo/_aux/_theme_oce.r")

root <- "/Users/pablogguz_/Documents/GitHub/poli-io-new/pablo/"

# Load MPD data
data_mpd <- read_dta(paste0(root, "output/tables/MPD_overtime_charts_pol.dta"))

# Convert labelled variables
data_mpd <- data_mpd %>%
    mutate(across(where(haven::is.labelled), ~ {
        label <- attr(.x, "label")
        result <- as.numeric(.x)
        attr(result, "label") <- label
        result
    }))

# Define variables and labels
variables_to_plot <- c("p82_idx_econ", "p82_idx_cult")

variable_labels <- c(
    "p82_idx_econ" = "Economic",
    "p82_idx_cult" = "Cultural"
)

# Prepare plot data
plot_data <- data_mpd %>%
    select(election_period, all_of(variables_to_plot)) %>%
    pivot_longer(cols = all_of(variables_to_plot), names_to = "variable", values_to = "value") %>%
    group_by(variable) %>%
    mutate(
        # Index to first period = 100
        value = (value / first(value)) * 100,
        period_label = case_when(
            election_period == 1 ~ "1995-1999",
            election_period == 2 ~ "2000-2004",
            election_period == 3 ~ "2005-2009",
            election_period == 4 ~ "2010-2014",
            election_period == 5 ~ "2015-2019",
            TRUE ~ as.character(election_period)
        ),
        period_label = factor(period_label, levels = c(
            "1995-1999", "2000-2004", "2005-2009",
            "2010-2014", "2015-2019"
        )),
        variable_label = variable_labels[variable]
    ) %>%
    ungroup()

# Label data for last point
label_data <- plot_data %>%
    filter(election_period == max(election_period)) %>%
    mutate(label_text = as.character(round(value, 0)))

# Create plot
plot <- ggplot(plot_data, aes(x = period_label, y = value, group = variable_label, color = variable_label)) +
    geom_hline(yintercept = 100, linetype = "dashed", color = "grey50", linewidth = 0.8) +
    geom_line(linewidth = 2) +
    geom_point(size = 4, shape = 21, fill = "white", stroke = 1.5) +
    geom_text_repel(
        data = label_data,
        aes(label = label_text),
        size = 6,
        fontface = "bold",
        point.padding = 0.3,
        box.padding = 0.4,
        force = 1,
        direction = "y",
        hjust = 0,
        nudge_x = 0.1,
        segment.size = 0.3,
        segment.color = "grey50",
        min.segment.length = 0,
        bg.color = "white",
        bg.r = 0.15,
        show.legend = FALSE,
        family = "Libre Franklin"
    ) +
    scale_x_discrete(name = "") +
    scale_y_continuous(
        name = "Average within-country difference between\n80th and 20th percentile (1995-1999 = 100)",
        labels = scales::label_number(accuracy = 1),
        breaks = seq(60, 180, by = 20),
        limits = c(60, 180)
    ) +
    scale_color_manual(
        name = "",
        values = c(
            "Economic" = "#E31A1C",
            "Cultural" = "#1F78B4"
        )
    ) +
    theme_oce(base_size = 18) +
    theme(
        text = element_text(family = "Libre Franklin"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size = 18),
        axis.title = element_text(size = 18),
        legend.position = "top",
        legend.text = element_text(size = 18),
        plot.caption = element_textbox_simple(size = 11, hjust = 0, color = "grey40")
    ) +
    guides(color = guide_legend(override.aes = list(size = 3), nrow = 1))

# Save
dir.create(paste0(root, "output/figures/mpd_charts_overttime"), recursive = TRUE, showWarnings = FALSE)
ggsave(paste0(root, "output/figures/mpd_charts_overttime/mpd_p82_combined.png"), plot, width = 12, height = 6.5, dpi = 150)

# Export raw data to Excel
export_data <- data_mpd %>%
    select(election_period, all_of(variables_to_plot)) %>%
    mutate(
        `Election year` = case_when(
            election_period == 1 ~ "1995-1999",
            election_period == 2 ~ "2000-2004",
            election_period == 3 ~ "2005-2009",
            election_period == 4 ~ "2010-2014",
            election_period == 5 ~ "2015-2019",
            TRUE ~ as.character(election_period)
        ),
        across(
            starts_with("p82_idx_"),
            ~ (.x / first(.x)) * 100
        )
    ) %>%
    select(
        `Election year`,
        `Economic` = p82_idx_econ,
        `Cultural` = p82_idx_cult
    )

writexl::write_xlsx(export_data, paste0(root, "output/figures/mpd_charts_overttime/mpd_p82_combined_raw_data.xlsx"))
cat("Done.\n")

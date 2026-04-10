#' EBRD Office of the Chief Economist (OCE) ggplot2 Theme and Color Palettes
#' 
#' This script provides a custom ggplot2 theme and color palettes for the EBRD OCE.
#' 
#' Author: Pablo Garcia Guzman
#' Last updated: November 2024

library(ggplot2)

# Base EBRD Colors for reference
ebrd_blue <- "#00448D"
ebrd_orange <- "#ffad41"
ebrd_red <- "#E41E25"  # Adding red for diverging palette

#' EBRD OCE Color Palettes
#' Simplified set of palettes for different visualization needs
palettes_oce <- list(
  # Main multicolor palette with 8 distinct colors
  multicolor = c(
    "#00448D",  # EBRD blue
    "#ffad41",  # EBRD orange
    "#8DB4E2",  # Light blue
    "#58585A",  # Grey
    "#FFD599",  # Light orange
    "#002B59",  # Darker blue
    "#B47A30",  # Darker orange
    "#A6CEF7"   # Lighter blue
  ),
  
  # Gradient blue (5 blues from light to dark)
  gradient_blue = colorRampPalette(c("#FFFFFF", "#8DB4E2", "#00448D")),
  
  # Diverging blue to red
  diverging = colorRampPalette(c("#00448D", "#FFFFFF", "#E41E25"))
)

#' Function to retrieve EBRD OCE palettes
#' 
#' @param name Name of the palette ("multicolor", "gradient_blue", "diverging")
#' @param n Number of colors to return (required for gradient palettes)
#' @return A vector of colors
palette_oce <- function(name, n = NULL) {
  if (name %in% c("gradient_blue", "diverging")) {
    if (is.null(n)) stop("Number of colors needed for gradient palettes")
    return(palettes_oce[[name]](n))
  }
  
  return(palettes_oce[[name]])
}

#' Function to retrieve EBRD OCE palettes
#' 
#' @param name Name of the palette ("main", "three_colors", "blues", "oranges", "contrast", "full")
#' @param n Number of colors to return
#' @return A vector of colors
palette_oce <- function(name, n = NULL) {
  pal <- palettes_oce[[name]]
  
  # For gradient palettes (which are functions)
  if (is.function(pal)) {
    if (is.null(n)) stop("Number of colors (n) must be provided for gradient palettes")
    return(pal(n))
  }
  
  # For discrete palettes
  if (!is.null(n)) {
    if (n > length(pal)) {
      stop("Requested number of colors (", n, 
           ") is greater than the palette length (", length(pal), ")")
    }
    return(pal[1:n])
  }
  
  return(pal)
}

#' Print function for color palettes
#' 
#' @param x Palette to print
#' @param ... Additional arguments
print.palette <- function(x, ...) {
  n <- length(x)
  old <- par(mar = c(0.5, 0.5, 0.5, 0.5))
  on.exit(par(old))
  
  image(1:n, 1, as.matrix(1:n), col = x,
        ylab = "", xaxt = "n", yaxt = "n", bty = "n")
  rect(0, 0.9, n + 1, 1.1, col = rgb(1, 1, 1, 0.8), border = NA)
  text((n + 1) / 2, 1, labels = attr(x, "name"), cex = 1, family = "serif")
}

#' EBRD OCE ggplot2 theme
#' 
#' @param base_size Base font size
#' @param base_family Base font family (defaults to Franklin Gothic Book)
#' @return A ggplot2 theme
theme_oce <- function(base_size = 16, base_family = "Libre Franklin Regular") {
  theme_minimal(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # Text elements with explicit black color
      text = element_text(family = base_family, size = base_size, color = "black"),
      axis.text = element_text(color = "black"),
      axis.title = element_text(size = base_size, color = "black"),
      legend.text = element_text(size = base_size, color = "black"),
      strip.text = element_text(size = base_size, color = "black"),
      
      # Grid elements
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      
      # Legend position and formatting
      legend.position = "top",
      legend.justification = "left",
      legend.direction = "horizontal",
      legend.title = element_blank(),
      
      # Facet strip placement - outside and below
      strip.placement = "outside",
      strip.text.x = element_text(margin = margin(t = 5, b = 5)), # Add some margin
      panel.spacing = unit(2, "lines"), # Add some space between facets
      
      # Background elements
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      strip.background = element_rect(fill = "white", color = NA),
      
      # Title elements
      plot.title = element_textbox_simple(
        size = base_size * 1.2, 
        face = "bold", 
        color = "black", 
        hjust = 0, 
        vjust = 1
      ),
      
      # Caption 
      plot.caption = element_textbox_simple(),

      # Complete theme
      complete = TRUE
    )
}

# Jarvis Trading Bot - R Analysis Suite

library(tidyverse)
library(tidyquant)

# Función para cargar el estado de la cartera
load_portfolio <- function() {
  if (file.exists("data/portfolio_state.csv")) {
    read_csv("data/portfolio_state.csv")
  } else {
    message("Archivo de cartera no encontrado.")
  }
}

# Función para visualizar el historial de precios generado por el bot
plot_market_history <- function() {
  if (file.exists("data/history.csv")) {
    history <- read_csv("data/history.csv")
    history %>%
      ggplot(aes(x = date, y = close, color = symbol)) +
      geom_line() +
      theme_minimal() +
      labs(title = "Análisis de Mercado - Jarvis Trading Bot",
           subtitle = "Datos exportados para validación estadística en R",
           x = "Fecha", y = "Precio de Cierre (USD)")
  }
}

# Aquí Luis puede realizar pruebas de hipótesis sobre las decisiones del bot

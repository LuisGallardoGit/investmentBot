# Jarvis Trading Bot - Monte Carlo Simulation & Validation

library(tidyverse)

#' Simulación de Monte Carlo para validación de estrategias
#' @param initial_equity Capital inicial
#' @param expected_return Retorno diario esperado
#' @param volatility Volatilidad diaria (desviación estándar)
#' @param days Número de días a simular
#' @param simulations Número de trayectorias
simulate_monte_carlo <- function(initial_equity = 10000, expected_return = 0.0005, 
                                volatility = 0.015, days = 60, simulations = 1000) {
  
  sim_matrix <- matrix(nrow = days, ncol = simulations)
  
  for(i in 1:simulations) {
    daily_returns <- rnorm(days, mean = expected_return, sd = volatility)
    sim_matrix[,i] <- initial_equity * cumprod(1 + daily_returns)
  }
  
  return(as.data.frame(sim_matrix))
}

#' Visualizar resultados de simulación
plot_monte_carlo <- function(sim_data) {
  sim_data %>%
    mutate(day = 1:n()) %>%
    pivot_longer(-day, names_to = "simulation", values_to = "equity") %>%
    ggplot(aes(x = day, y = equity, group = simulation)) +
    geom_line(alpha = 0.1, color = "steelblue") +
    geom_hline(yintercept = 10000, linetype = "dashed", color = "red") +
    theme_minimal() +
    labs(title = "Simulación de Monte Carlo (1,000 escenarios)",
         subtitle = "Validación de supervivencia de la estrategia a 60 días",
         x = "Días de Operación", y = "Equity (USD)")
}

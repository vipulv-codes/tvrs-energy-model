# TVRS Model for Energy Futures

This repository contains a complete Python implementation of the Time-Varying Volatility Regime Switching (TVRS) model for pricing European options on energy futures, as described in the paper:
*"Time-varying volatility model equipped with regime switching factor: Valuation of option price written on energy futures"* by Guillaume Leduc, Farshid Mehrdoust, and Idin Noorani.

## Features

- **Semi-Analytical Pricing**: Prices European call options using the Fourier Transform method (Theorem 3.1).
- **Hedging Strategy**: Computes the required futures position (Delta) and money market position to form a mean-self-financing portfolio (Theorem 3.2).
- **Monte Carlo Simulation (GAVE)**: Implements the Generalized Antithetic Variates Estimator to robustly cross-verify the analytical option prices.
- **EM Calibration**: An Expectation-Maximization algorithm utilizing Genetic Optimization to calibrate model parameters ($\kappa, \xi_1, \xi_2, \lambda_1, \lambda_2$) from historical daily futures data.

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To see an end-to-end demonstration (fetching real data, pricing analytically, comparing with Monte Carlo, and outputting the hedging strategy), simply run:

```bash
python example_usage.py
```

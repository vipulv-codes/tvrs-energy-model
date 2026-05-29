import numpy as np
from tvrs_model.pricing import TVRSModel
from tvrs_model.simulation import TVRSSimulation
from tvrs_model.calibration import EMCalibration, extract_intensities, download_data

def main():
    print("=== TVRS Model for Energy Futures ===")
    
    # 1. Fetching Real Data & Calibration
    print("\n[1] Fetching historical data and running EM Calibration...")
    try:
        # We download data to demonstrate functionality, but we will use the parameters from the paper 
        # to ensure the results match the paper for the rest of the script.
        prices = download_data()
        print(f"Downloaded {len(prices)} days of NG=F prices.")
        
        # Uncomment the following to run the full calibration (can be slow due to Genetic Algorithm)
        # delta = 0.0455  # Granularity from paper (~1/22 trading days per month)
        #
        # Option A — Scalar T_minus_t (acceptable for rolling front-month NG=F,
        # where time-to-maturity is roughly constant at ~1 month = 1/12 year):
        # T_minus_t = 1.0 / 12
        # em = EMCalibration(prices[:500], delta, T_minus_t)
        #
        # Option B — Array T_minus_t (strict paper implementation; use when you
        # have the exact maturity date of each contract in the dataset):
        # n_obs = 500
        # T_contract = 2.0  # e.g. 2-year contract
        # t_obs = np.arange(n_obs) * delta
        # T_minus_t_array = T_contract - t_obs  # tau decreases with each observation
        # em = EMCalibration(prices[:n_obs], delta, T_minus_t_array)
        #
        # params, p_trans, smooth_prob = em.fit(max_iter=20)
        # kappa_cal, xi1_cal, xi2_cal = params
        #
        # Convert discrete transition matrix to continuous intensities for pricing:
        # lambda1_cal, lambda2_cal, Pi = extract_intensities(p_trans, delta)
        # print(f"Calibrated: kappa={kappa_cal:.4f}, xi1={xi1_cal:.4f}, xi2={xi2_cal:.4f}")
        # print(f"Extracted:  lambda1={lambda1_cal:.4f}, lambda2={lambda2_cal:.4f}")
        
        print("Data fetched successfully. Using parameters from Table 1 for pricing.")
    except Exception as e:
        print(f"Could not download data: {e}")

    # Parameters from Table 1 in the paper
    kappa = 0.0054
    xi1 = 1.1275
    xi2 = 0.8700
    lambda1 = 2.3055
    lambda2 = 12.4695
    r1 = 0.02
    r2 = 0.04
    
    # Current futures price and contract details
    f = 3.5  # Current futures price
    K = 3.0  # Strike price
    t = 0.0  # Current time
    T = 1.0  # Time to maturity
    initial_state = np.array([1, 0]) # State 1
    
    print("\n[2] Calculating European Call Option Price (Analytical)")
    pricing_model = TVRSModel(kappa, xi1, xi2, lambda1, lambda2, r1, r2, alpha_damp=1.5)
    
    price_analytical = pricing_model.price_european_call(f, K, t, T, initial_state)
    print(f"Analytical Option Price (Theorem 3.1): {price_analytical:.4f}")
    
    print("\n[3] Calculating European Call Option Price (GAVE Monte Carlo)")
    sim_model = TVRSSimulation(kappa, xi1, xi2, lambda1, lambda2, r1, r2)
    # Using small m and n for quick execution. Increase m and n for accurate results.
    price_gave = sim_model.price_european_call(f, K, T, n=100, d=16, m=1000, initial_state=1)
    print(f"GAVE Simulation Price (Algorithm 3): {price_gave:.4f}")
    
    abs_error = abs(price_analytical - price_gave)
    rel_error = abs_error / price_gave if price_gave > 0 else 0
    print(f"Absolute Error: {abs_error:.4f}")
    print(f"Relative Error: {rel_error:.2%}")
    
    print("\n[4] Calculating Hedging Strategy")
    gamma_t, beta_t = pricing_model.hedging_strategy(f, K, t, T, initial_state)
    print(f"Number of futures held (gamma_t): {gamma_t:.4f}")
    print(f"Money market position (beta_t):   {beta_t:.4f}")
    print(f"(Note: beta_t = option price for futures; see Theorem 3.2)")
    
    print("\nDone.")

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import yfinance as yf
from scipy.stats import norm

class EMCalibration:
    def __init__(self, data, delta, T_minus_t):
        """
        data: array of prices F_t
        delta: time step size
        T_minus_t: time to maturity, either:
            - scalar: treated as constant tau for all observations (e.g. rolling
              front-month where maturity lag is approximately fixed)
            - 1-D array of length len(data): tau_j = T - t_j for each observation,
              which correctly captures the paper's time-varying sigma(t, T, X_t)
        """
        self.F = data
        self.delta = delta
        # Normalise to a per-observation array so downstream code is uniform
        if np.isscalar(T_minus_t):
            self.T_minus_t = np.full(len(data), T_minus_t)
        else:
            self.T_minus_t = np.asarray(T_minus_t, dtype=float)
            if len(self.T_minus_t) != len(data):
                raise ValueError(
                    f"T_minus_t array length ({len(self.T_minus_t)}) must match "
                    f"data length ({len(data)})"
                )
        self.n = len(data) - 1

    def _sigma(self, kappa, xi, tau):
        if kappa == 0:
            return xi
        # Handling the case where tau is an array or scalar
        return xi * np.sqrt((1 - np.exp(-2 * kappa * tau)) / (2 * kappa * tau))

    def _G(self, F_curr, F_prev, sigma):
        """Gaussian transition density."""
        var = (sigma * F_prev)**2 * self.delta
        std = np.sqrt(var)
        # To avoid division by zero or log(0)
        std = np.maximum(std, 1e-8)
        return norm.pdf(F_curr, loc=F_prev, scale=std)

    def e_step(self, params, p_trans):
        """
        params: (kappa, xi1, xi2)
        p_trans: 2x2 transition matrix P(X_t = j | X_{t-1} = i)
        """
        kappa, xi1, xi2 = params
        
        # Filtering
        # filter_prob[j, k] = P(X_{t_j} = e_k | F_{t_0}, ..., F_{t_j})
        filter_prob = np.zeros((self.n + 1, 2))
        filter_prob[0, :] = [0.5, 0.5] # initial guess
        
        predict_prob = np.zeros((self.n + 1, 2))
        
        for j in range(1, self.n + 1):
            # Predict
            predict_prob[j, 0] = filter_prob[j-1, 0] * p_trans[0, 0] + filter_prob[j-1, 1] * p_trans[1, 0]
            predict_prob[j, 1] = filter_prob[j-1, 0] * p_trans[0, 1] + filter_prob[j-1, 1] * p_trans[1, 1]
            
            # sigma is evaluated at the current observation time t_j using tau_j = T - t_j
            # This gives the paper's time-varying sigma(t_j, T, X_{t_j})
            tau_j = self.T_minus_t[j]
            sigma1 = self._sigma(kappa, xi1, tau_j)
            sigma2 = self._sigma(kappa, xi2, tau_j)
            
            # Update (filtering)
            G1 = self._G(self.F[j], self.F[j-1], sigma1)
            G2 = self._G(self.F[j], self.F[j-1], sigma2)
            
            num1 = predict_prob[j, 0] * G1
            num2 = predict_prob[j, 1] * G2
            denom = num1 + num2
            
            if denom == 0:
                filter_prob[j, :] = [0.5, 0.5]
            else:
                filter_prob[j, 0] = num1 / denom
                filter_prob[j, 1] = num2 / denom
                
        # Smoothing
        # smooth_prob[j, k] = P(X_{t_j} = e_k | F_{t_0}, ..., F_{t_n})
        smooth_prob = np.zeros((self.n + 1, 2))
        smooth_prob[self.n, :] = filter_prob[self.n, :]
        
        for j in range(self.n - 1, -1, -1):
            for k in range(2):
                sum_term = 0
                for i in range(2):
                    if predict_prob[j+1, i] > 0:
                        sum_term += smooth_prob[j+1, i] * (filter_prob[j, k] * p_trans[k, i]) / predict_prob[j+1, i]
                smooth_prob[j, k] = sum_term
                
        return filter_prob, smooth_prob

    def m_step_obj(self, params, smooth_prob):
        """
        Objective function for the M-step GA optimizer (Negative Log-Likelihood)
        params: (kappa, xi1, xi2)
        """
        kappa, xi1, xi2 = params
        
        log_like = 0
        for j in range(1, self.n + 1):
            # Use per-observation tau to keep sigma time-varying, matching e_step
            tau_j = self.T_minus_t[j]
            sigma1 = self._sigma(kappa, xi1, tau_j)
            sigma2 = self._sigma(kappa, xi2, tau_j)
            
            G1 = self._G(self.F[j], self.F[j-1], sigma1)
            G2 = self._G(self.F[j], self.F[j-1], sigma2)
            
            log_G1 = np.log(max(G1, 1e-100))
            log_G2 = np.log(max(G2, 1e-100))
            
            log_like += smooth_prob[j, 0] * log_G1 + smooth_prob[j, 1] * log_G2
            
        return -log_like # Minimizing negative log-likelihood

    def fit(self, max_iter=20, tol=1e-3):
        # Initial guess from Table 1
        params = [0.0054, 1.1275, 0.8700] # kappa, xi1, xi2
        p_trans = np.array([[0.9236, 0.0764], [0.4131, 0.5869]])
        
        bounds = [(0.0001, 1.0), (0.1, 5.0), (0.1, 5.0)] # Bounds for kappa, xi1, xi2
        
        for iteration in range(max_iter):
            print(f"EM Iteration {iteration+1}/{max_iter}")
            
            # E-Step
            filter_prob, smooth_prob = self.e_step(params, p_trans)
            
            # Update transition probabilities
            # p_ki = sum_{j=1}^{n-1} P(X_{t_{j+1}} = i | F) * P(X_{t_j} = k | F) ... simplified estimation
            # A common estimation for transition matrix using smooth probs:
            p_trans_new = np.zeros((2, 2))
            for k in range(2):
                denom = np.sum(smooth_prob[:-1, k])
                if denom > 0:
                    for i in range(2):
                        # Approximate joint probability P(X_t=k, X_{t+1}=i | F)
                        num = 0
                        for j in range(self.n):
                            predict_prob_ji = filter_prob[j, 0] * p_trans[0, i] + filter_prob[j, 1] * p_trans[1, i]
                            if predict_prob_ji > 0:
                                joint = smooth_prob[j+1, i] * filter_prob[j, k] * p_trans[k, i] / predict_prob_ji
                                num += joint
                        p_trans_new[k, i] = num / denom
                else:
                    p_trans_new[k, :] = p_trans[k, :]
                    
            p_trans = p_trans_new
            p_trans = p_trans / p_trans.sum(axis=1, keepdims=True) # Normalize
            
            # M-Step
            res = differential_evolution(self.m_step_obj, bounds, args=(smooth_prob,), strategy='best1bin', maxiter=20)
            new_params = res.x
            
            # Check convergence
            mse = np.mean((np.array(params) - np.array(new_params))**2)
            params = new_params
            
            print(f"Params: kappa={params[0]:.4f}, xi1={params[1]:.4f}, xi2={params[2]:.4f}, MSE={mse:.6f}")
            if mse < tol:
                print("Converged!")
                break
                
        return params, p_trans, smooth_prob

def download_data():
    """Download US Natural Gas Futures data."""
    print("Downloading US Natural Gas Futures (NG=F) from Yahoo Finance...")
    # Jan 2017 to Jan 2023
    ng_data = yf.download("NG=F", start="2017-01-01", end="2023-01-01")
    # Drop NaN rows that yfinance can return for holidays / missing feed data;
    # passing NaNs into the EM filter produces NaN densities that break optimization.
    prices = ng_data['Close'].dropna().values
    return prices

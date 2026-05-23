import numpy as np
import scipy.integrate as integrate
import scipy.linalg as linalg

class TVRSModel:
    def __init__(self, kappa, xi1, xi2, lambda1, lambda2, r1, r2, alpha_damp=1.5):
        self.kappa = kappa
        self.xi1 = xi1
        self.xi2 = xi2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.r1 = r1
        self.r2 = r2
        self.alpha_damp = alpha_damp

    def _V(self, t, T, xi):
        """Computes the integral of sigma^2(s) from t to T."""
        if self.kappa == 0:
            # Limit as kappa -> 0 of (1 - exp(-2*kappa*(T-s))) / (2*kappa*(T-s)) is 1
            return xi**2 * (T - t)
        
        def integrand(s):
            tau = T - s
            if tau == 0:
                return xi**2
            return xi**2 * (1 - np.exp(-2 * self.kappa * tau)) / (2 * self.kappa * tau)
        
        val, _ = integrate.quad(integrand, t, T)
        return val

    def psi(self, v, t, T, x):
        """Computes the psi function for the Fourier transform."""
        z = v - 1j * (self.alpha_damp + 1)
        
        V1 = self._V(t, T, self.xi1)
        V2 = self._V(t, T, self.xi2)
        
        tau = T - t
        G1 = -self.r1 * tau - 0.5 * (1j * z + z**2) * V1
        G2 = -self.r2 * tau - 0.5 * (1j * z + z**2) * V2
        
        # Matrix M = diag(G) + Pi * tau
        M = np.array([
            [G1 - self.lambda1 * tau, self.lambda1 * tau],
            [self.lambda2 * tau, G2 - self.lambda2 * tau]
        ])
        
        expM = linalg.expm(M)
        
        # x is the initial state, e.g., np.array([1, 0]) for state 1
        num = np.dot(x, np.dot(expM, np.array([1, 1])))
        den = (1 + self.alpha_damp + 1j * v) * (self.alpha_damp + 1j * v)
        
        return num / den

    def price_european_call(self, f, K, t, T, x):
        """
        Prices a European call option.
        f: futures price
        K: strike price
        t: current time
        T: maturity
        x: initial state vector, e.g., np.array([1, 0]) for regime 1, or [0, 1] for regime 2.
        """
        if t == T:
            return max(f - K, 0)
            
        k = np.log(K / f)
        
        def integrand(v):
            psi_val = self.psi(v, t, T, x)
            return np.real(np.exp(-1j * v * k) * psi_val)
            
        # Integrate from 0 to infinity
        integral_val, _ = integrate.quad(integrand, 0, 100, limit=200) # 100 is usually enough for convergence
        
        price = (f * np.exp(-self.alpha_damp * k) / np.pi) * integral_val
        return price

    def hedging_strategy(self, f, K, t, T, x):
        """
        Computes the delta and money market positions for hedging.
        Returns (gamma_t, beta_t)
        """
        eps = f * 1e-4
        
        # Finite difference for Delta (gamma_t)
        U_up = self.price_european_call(f + eps, K, t, T, x)
        U_down = self.price_european_call(f - eps, K, t, T, x)
        
        gamma_t = (U_up - U_down) / (2 * eps)
        
        # Option price
        U = self.price_european_call(f, K, t, T, x)
        
        # Beta_t calculation: beta_t = U - gamma_t * f
        beta_t = U - gamma_t * f
        
        return gamma_t, beta_t

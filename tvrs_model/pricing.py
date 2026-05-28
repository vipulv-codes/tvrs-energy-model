import numpy as np
import scipy.integrate as integrate
import scipy.linalg as linalg

class TVRSModel:
    def __init__(self, kappa, xi1, xi2, lambda1, lambda2, r1, r2, alpha_damp=1.5, integration_limit=100.0):
        self.kappa = kappa
        self.xi1 = xi1
        self.xi2 = xi2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.r1 = r1
        self.r2 = r2
        self.alpha_damp = alpha_damp
        self.integration_limit = integration_limit

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
            
        # Integrate from 0 to infinity using the tunable limit
        integral_val, _ = integrate.quad(integrand, 0, self.integration_limit, limit=200)
        
        price = (f * np.exp(-self.alpha_damp * k) / np.pi) * integral_val
        return price

    def hedging_strategy(self, f, K, t, T, x):
        """
        Computes the delta and money market positions for hedging a futures option.

        Unlike stock options, a futures contract costs ZERO to enter — only price
        changes dF_t generate P&L. The self-financing condition is therefore:
            dV_t = gamma_t * dF_t + r * beta_t * dt
        with portfolio value V_t = beta_t (money market only).

        To replicate the option at time t:
            - Hold gamma_t futures contracts  (zero cost to enter)
            - Invest beta_t = U(f, x, t) in the money market  (entire option value)

        NOTE: beta_t = U, NOT U - gamma_t * f. The subtraction of gamma_t * f
        is the stock convention (where you pay f upfront), which is incorrect here.

        Returns:
            gamma_t : number of futures contracts to hold (Delta)
            beta_t  : amount to invest in the money market (= option price)
        """
        eps = f * 1e-4

        # Finite difference for Delta (gamma_t) — unchanged by futures convention
        U_up = self.price_european_call(f + eps, K, t, T, x)
        U_down = self.price_european_call(f - eps, K, t, T, x)

        gamma_t = (U_up - U_down) / (2 * eps)

        # Option price = money market position (futures convention: no upfront cost)
        beta_t = self.price_european_call(f, K, t, T, x)

        return gamma_t, beta_t

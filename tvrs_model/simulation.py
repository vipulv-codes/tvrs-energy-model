import numpy as np

class TVRSSimulation:
    def __init__(self, kappa, xi1, xi2, lambda1, lambda2, r1, r2):
        self.kappa = kappa
        self.xi1 = xi1
        self.xi2 = xi2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.r1 = r1
        self.r2 = r2

    def _sigma(self, t, T, xi):
        """Volatility function"""
        tau = T - t
        if tau == 0:
            return xi
        if self.kappa == 0:
            return xi
        return xi * np.sqrt((1 - np.exp(-2 * self.kappa * tau)) / (2 * self.kappa * tau))

    def simulate_path(self, f, T, n, d, m, initial_state):
        """
        Simulate futures price paths using GAVE.
        f: initial price
        T: maturity
        n: time steps
        d: number of antithetic variates
        m: number of Monte Carlo paths
        initial_state: 1 or 2
        
        Returns: option_payoffs, discounted_factors
        """
        delta = T / n
        t_grid = np.linspace(0, T, n + 1)
        
        F = np.full((m, d), f)
        discount_factor = np.ones((m, d))
        
        for p in range(m):
            # Algorithm 1: Simulate state changes for this path
            t_current = 0
            state = initial_state
            
            states = np.zeros(n)
            
            # Draw holding times
            regime_times = []
            regime_states = []
            
            while t_current < T:
                lam = self.lambda1 if state == 1 else self.lambda2
                hold_time = np.random.exponential(1 / lam) if lam > 0 else T
                
                t_next = min(t_current + hold_time, T)
                regime_times.append((t_current, t_next))
                regime_states.append(state)
                
                t_current = t_next
                state = 2 if state == 1 else 1
            
            # Map continuous time regimes to discrete grid
            r_path = np.zeros(n)
            xi_path = np.zeros(n)
            
            for i in range(n):
                t_mid = t_grid[i] + delta / 2
                # find which regime t_mid falls into
                for (start, end), s in zip(regime_times, regime_states):
                    if start <= t_mid <= end:
                        r_path[i] = self.r1 if s == 1 else self.r2
                        xi_path[i] = self.xi1 if s == 1 else self.xi2
                        break
            
            # Simulate F path
            for i in range(n):
                t_i = t_grid[i]
                r_i = r_path[i]
                xi_i = xi_path[i]
                
                sigma_i = self._sigma(t_i, T, xi_i)
                
                # Generate d standard normals with correlation -1/(d-1)
                Z = np.random.randn(d)
                Z_mean = np.mean(Z)
                y = np.sqrt(d / (d - 1)) * (Z - Z_mean) if d > 1 else np.random.randn(1)
                
                # Log-Euler step (to avoid negative prices)
                F[p, :] = F[p, :] * np.exp(-0.5 * sigma_i**2 * delta + sigma_i * np.sqrt(delta) * y)
                
                # Update discount factor
                discount_factor[p, :] *= np.exp(-r_i * delta)
                
        return F, discount_factor

    def price_european_call(self, f, K, T, n=100, d=16, m=1000, initial_state=1):
        """
        Prices a European call option using GAVE method.
        """
        if T == 0:
            return max(f - K, 0)
            
        F_final, discount_factor = self.simulate_path(f, T, n, d, m, initial_state)
        
        # Calculate payoff for each branch
        payoffs = np.maximum(F_final - K, 0)
        
        # Multiply by discount factor
        discounted_payoffs = discount_factor * payoffs
        
        # Price is the mean over all m paths and d branches
        price = np.mean(discounted_payoffs)
        
        return price

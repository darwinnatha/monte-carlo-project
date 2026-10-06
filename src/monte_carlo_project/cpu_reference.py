import numpy as np
from scipy.stats import norm

def black_scholes_call(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calcule le prix exact d'une option d'achat (Call) européenne via Black-Scholes.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def monte_carlo_pi_cpu(N: int) -> float:
    """
    Estimation de Pi sur CPU via Monte-Carlo vectorisé.
    """
    x = np.random.uniform(-1.0, 1.0, N)
    y = np.random.uniform(-1.0, 1.0, N)
    inside = (x**2 + y**2) <= 1.0
    return 4.0 * np.sum(inside) / N

def monte_carlo_option_cpu(S0: float, K: float, T: float, r: float, sigma: float, N: int) -> tuple[float, float]:
    """
    Valorisation d'une option européenne sur CPU par Mouvement Brownien Géométrique.
    Retourne le prix estimé et l'erreur standard de la simulation.
    """
    Z = np.random.standard_normal(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)
    discounted_payoffs = np.exp(-r * T) * payoffs
    
    price = np.mean(discounted_payoffs)
    std_err = np.std(discounted_payoffs) / np.sqrt(N)
    return float(price), float(std_err)

def metropolis_hastings_cpu(target_pdf, x0: float, n_samples: int, proposal_std: float = 0.5) -> np.ndarray:
    """
    Algorithme de Metropolis-Hastings à une chaîne sur CPU.
    - target_pdf : fonction proportionnelle à la densité cible P(x)
    - x0 : point de départ de la chaîne
    - n_samples : nombre de tirages
    """
    samples = np.empty(n_samples, dtype=np.float64)
    x_current = x0
    
    for i in range(n_samples):
        # Proposition d'un nouvel état via une loi normale symétrique
        x_proposal = np.random.normal(x_current, proposal_std)
        
        # Calcul du ratio d'acceptation alpha
        p_current = target_pdf(x_current)
        p_proposal = target_pdf(x_proposal)
        alpha = min(1.0, p_proposal / p_current if p_current > 0 else 1.0)
        
        # Acceptation ou rejet
        if np.random.rand() < alpha:
            x_current = x_proposal
            
        samples[i] = x_current
        
    return samples
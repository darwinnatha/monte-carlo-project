import os
import math
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64

# S'assurer que le mode simulateur est documenté/activé pour le dev local
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

@cuda.jit
def monte_carlo_option_kernel(rng_states, S0, K, T, r, sigma, N_per_thread, d_payoffs):
    """
    Noyau CUDA : chaque thread simule N_per_thread trajectoires
    du Mouvement Brownien Géométrique et accumule les payoffs actualisés.
    """
    idx = cuda.grid(1)
    if idx < d_payoffs.shape[0]:
        accumulated_payoff = 0.0
        discount_factor = math.exp(-r * T)
        drift = (r - 0.5 * sigma * sigma) * T
        vol_sqrtT = sigma * math.sqrt(T)

        for _ in range(N_per_thread):
            # Génération d'une variable normale centrée réduite ~ N(0, 1)
            z = xoroshiro128p_normal_float64(rng_states, idx)
            ST = S0 * math.exp(drift + vol_sqrtT * z)
            payoff = max(ST - K, 0.0)
            accumulated_payoff += payoff

        d_payoffs[idx] = (accumulated_payoff / N_per_thread) * discount_factor

def run_monte_carlo_option_gpu(S0, K, T, r, sigma, N_total=100000, threads_per_block=128):
    """
    Fonction hôte pour orchestrer l'exécution du noyau GPU.
    """
    blocks = 64
    total_threads = blocks * threads_per_block
    N_per_thread = math.ceil(N_total / total_threads)

    # Initialisation des états du générateur pseudo-aléatoire
    rng_states = create_xoroshiro128p_states(total_threads, seed=42)
    d_payoffs = cuda.device_array(total_threads, dtype=np.float64)

    # Lancement du noyau GPU
    monte_carlo_option_kernel[blocks, threads_per_block](
        rng_states, S0, K, T, r, sigma, N_per_thread, d_payoffs
    )

    # Récupération et réduction finale sur l'hôte
    h_payoffs = d_payoffs.copy_to_host()
    return float(np.mean(h_payoffs))
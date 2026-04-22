"""
bernstein/basis.py

Base di Bernstein standard e funzioni di valutazione.

    B_{k,N}(x) = C(N,k) * x^k * (1-x)^{N-k}    per k = 0, ..., N

Poiché ogni base integra 1/(N+1), per BP(W,x) abbia integrale 1 serve:
    sum(W) = N+1

Le correzioni Delta soddisfano sum(Delta) = 0 per preservare la unit measure.

NB: viene usata la base così come è, non quella normalizzata.
Perciò ci sono delle differenze per quanto riguarda il vincolo di unit measure.
"""

import numpy as np
from scipy.special import comb


def basis_matrix(N: int, x: np.ndarray) -> np.ndarray:
    """
    Matrice della base di Bernstein standard di grado N.
        M[:, k] = C(N,k) * x^k * (1-x)^{N-k}
    Ogni colonna integra a 1/(N+1). Vincolo unit measure: sum(W) = N+1.
    Returns: M ndarray shape (len(x), N+1)
    """
    M = np.zeros((len(x), N + 1))
    for k in range(N + 1):
        M[:, k] = comb(N, k) * (x ** k) * ((1 - x) ** (N - k))
    return M


def eval_bp(W: np.ndarray, M: np.ndarray) -> np.ndarray:
    """ 
    Al posto di fare la sommatoria esplicita per trovare
    il valore del polinomio si fa il prodotto matriciale
    fra la base di bernstein ed i pesi (meglio computazionalmente)
    """
    return M @ W


# Costruisce la cdf in modo numerico a partire dalla pdf
def cdf_from_weights(W: np.ndarray, M: np.ndarray, dx: float) -> np.ndarray:
    pdf = np.maximum(M @ W, 0.0)
    cdf = np.cumsum(pdf) * dx
    cdf /= max(cdf[-1], 1e-12)
    return cdf


def bernstein_operator_init(N: int, x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Operatore di Bernstein classico: W_init[k] = f(k/N) con sum=N+1.
    Si parte da questi pesi per le ottimizzazioni successive (scipy e pytorch).
    In questo modo la condizione di unit measure è differente, essa è: sum(W) = N+1 e sum(Delta) = 0.
    """
    nodes = np.linspace(0, 1, N + 1)  # linspace -> crea N + 1 ascisse equidistanti fra 0 ed 1
    # crea un interpolante per la funzione nei nodi appena creati conosciuta la coppia (x, f(x))
    W = np.interp(nodes, x, f)
    W = np.maximum(W, 0.0)
    total = W.sum()
    W = W * (N + 1) / total if total > 1e-12 else np.ones(N + 1)  # riscalo
    return W


def mse(W: np.ndarray, M: np.ndarray, f: np.ndarray) -> float:
    return float(np.mean((f - M @ W) ** 2))

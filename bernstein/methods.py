"""
bernstein/methods.py

I quattro metodi di ottimizzazione dei pesi W del BP.

  A. solve_ing_cane          — soluzione quasi-chiusa, O(n)
  B. solve_scipy             — SLSQP vincolato con gradiente analitico
                               restituisce (W_new, Delta) con Delta = W_new - W_ref
  C. solve_pytorch           — gradient descent non vincolato (softmax)
  D. solve_pytorch_ordered   — gradient descent con penalità SO

Formulazione Delta:

Tutti i metodi con vincolo SO lavorano in termini di:

    Delta = W_new - W_ref

I vincoli mandatori sono:
  1) W_new[k] >= 0                        (positività)
  2) sum(W_new) = N+1                     (misura unitaria -> sum(Delta) = 0)
  non sum(robe) = 1 (perché è spiegato nel report_berstein.md)

I vincoli SO opzionali sono:
  3-upper) cumsum(Delta)[h] <= 0  ∀h      (massa verso destra)
  3-lower) cumsum(Delta)[h] >= 0  ∀h      (massa verso sinistra)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.optimize as scopt

from bernstein.basis import basis_matrix, bernstein_operator_init
from bernstein.stochastic import (build_scipy_constraints,
                                  stochastic_penalty_torch)


# A: Ing-Cane
def solve_ing_cane(N: int, x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Concentra tutta la massa sulle due basi di Bernstein che racchiudono
    la moda di f.

    Idea: B_{k,N} ha moda in k/N. La moda di f cade nell'intervallo
    [k/N, (k+1)/N]; si distribuisce la massa (N+1) tra B_{k,N} e B_{k+1,N}
    in proporzione inversa alla distanza dalla moda.

    Complessità: O(n) - frazioni di millisecondo.
    Limite: solo 2 basi attive su N+1 -> cattura il picco ma ignora la
            forma globale di f.
    """
    mode_x = x[np.argmax(f)]
    k = min(int(np.floor(mode_x * N)), N - 1)

    dist_right = (k + 1) / N - mode_x
    dist_left = mode_x - k / N
    interval = 1.0 / N

    W = np.zeros(N + 1)
    W[k] = dist_right / interval
    W[k + 1] = dist_left / interval
    W = W * (N + 1) / W.sum()
    return W


# B: Scipy SLSQP - ottimizzazione in spazio Delta
def solve_scipy(N: int, x: np.ndarray, f: np.ndarray,
                W_init: np.ndarray | None = None,
                direction: str | None = None,
                W_ref: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimizza MSE(W_ref + Delta) rispetto a Delta con SLSQP.

    La variabile di ottimizzazione è Delta = W_new - W_ref.
    W_new viene ricostruita come W_ref + Delta alla fine.

    Parametri:
    W_init    : punto di partenza per W_new (Delta_init = W_init - W_ref)
    direction : None | 'upper' | 'lower'
    W_ref     : pesi di riferimento (definiscono la soglia SO e il punto
                attorno a cui si muove Delta).
                Se None, usa W_init.

    Restituisce:
    (W_new, Delta)  con  Delta = W_new - W_ref,  sum(Delta) = 0

    Vincoli su Delta:
    Mandatori:
      1) W_ref[k] + Delta[k] >= 0        (positività di W_new)
         -> bounds: Delta[k] >= -W_ref[k]
      2) sum(Delta) = 0                   (misura unitaria, da sum(W_new)=N+1)

    SO opzionali:
      3-upper) cumsum(Delta)[h] <= 0  per h = 0..N-1
      3-lower) cumsum(Delta)[h] >= 0  per h = 0..N-1

    Gradiente analitico rispetto a Delta:
        MSE(W_ref + Delta) = mean((f - M(W_ref+Delta))²)
        grad_Delta MSE = -2/n · Mᵀ (f - M(W_ref+Delta))
                       = grad_W MSE  (identico, perché Delta appare linearmente)
    """
    M = basis_matrix(N, x)
    n_pts = len(x)

    if W_init is None:
        W_init = bernstein_operator_init(N, x, f)
    if W_ref is None:
        W_ref = W_init.copy()

    # Punto di partenza in spazio Delta
    Delta_init = W_init - W_ref

    def objective(Delta):
        W_new = W_ref + Delta
        r = f - M @ W_new
        return float(np.mean(r ** 2))

    def jac_obj(Delta):
        W_new = W_ref + Delta
        r = f - M @ W_new
        return -2.0 * (M.T @ r) / n_pts   # grad rispetto a Delta = grad rispetto a W

    # Vincoli SO puri rispetto a W_ref (threshold=0):
    #   upper: sum(Delta[0:h+1]) ≤ 0  per h=0..N-1
    #   lower: sum(Delta[h:])    ≤ 0  per h=1..N
    # Questo garantisce BP ≥_st W_ref (upper) o BP ≤_st W_ref (lower).
    # Per scipy_{upper,lower} W_ref=W_scipy≈f, quindi SO è rispetto a f.
    constraints = build_scipy_constraints(N, direction)

    # Bounds: Delta[k] >= -W_ref[k]  (garantisce W_new[k] >= 0)
    bounds = [(-W_ref[k], None) for k in range(N + 1)]

    result = scopt.minimize(
        objective, Delta_init,
        jac=jac_obj,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 20_000, 'ftol': 1e-13}
    )
    Delta = result.x
    W_new = np.maximum(W_ref + Delta, 0.0)
    Delta = W_new - W_ref   # ricalcola Delta coerente con il clamp
    return W_new, Delta


# C: PyTorch - gradient descent non vincolato
class _BernsteinModel(nn.Module):
    """
    W = softmax(W_raw) * (N+1)
    Garantisce W[k] > 0 e sum(W) = N+1 per costruzione.
    """

    def __init__(self, N: int):
        super().__init__()
        self.N = N
        self.W_raw = nn.Parameter(torch.zeros(N + 1))

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        W = torch.softmax(self.W_raw, dim=0) * (self.N + 1)
        return M @ W

    def weights(self) -> np.ndarray:
        with torch.no_grad():
            return (torch.softmax(self.W_raw, dim=0) * (self.N + 1)).numpy()


def solve_pytorch(N: int, x: np.ndarray, f: np.ndarray,
                  epochs: int = 3000, lr: float = 0.05) -> np.ndarray:
    """
    Gradient descent Adam puro: non applica vincoli di ordine stocastico.
    Utile come riferimento di accuratezza massima raggiungibile con GD.
    """
    M_t = torch.tensor(basis_matrix(N, x), dtype=torch.float32)
    f_t = torch.tensor(f, dtype=torch.float32)

    model = _BernsteinModel(N)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(M_t), f_t).backward()
        opt.step()

    return model.weights()


# D: PyTorch - gradient descent con penalità SO
def solve_pytorch_ordered(N: int, x: np.ndarray, f: np.ndarray,
                          W_ref: np.ndarray,
                          direction: str = 'upper',
                          epochs: int = 5000,
                          lr: float = 0.02,
                          lam: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient descent con penalità per violazioni dell'ordine stocastico:

        L(W) = MSE(W) + λ · Σ_h max(0, cumsum(Delta)[h] * segno)²

    dove Delta = W - W_ref e il segno dipende da direction.

    Nota: la penalità ammorbidisce il vincolo ma non lo garantisce esatto.

    Restituisce (W_new, Delta).
    """
    M_np = basis_matrix(N, x)
    M_t = torch.tensor(M_np, dtype=torch.float32)
    f_t = torch.tensor(f, dtype=torch.float32)
    Wr_t = torch.tensor(W_ref, dtype=torch.float32)

    # W_init: Pytorch unconstrained solution
    W_init = solve_pytorch(N, x, f)

    # softmax parametrization coerente con W_init
    w0 = torch.tensor(W_init / (N + 1), dtype=torch.float32)
    W_raw = nn.Parameter(torch.log(w0))
    opt = optim.Adam([W_raw], lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        W = torch.softmax(W_raw, dim=0) * (N + 1)
        mse_term = torch.mean((f_t - M_t @ W) ** 2)
        pen = lam * stochastic_penalty_torch(W, Wr_t, direction)
        (mse_term + pen).backward()
        opt.step()

    with torch.no_grad():
        W_new = (torch.softmax(W_raw, dim=0) * (N + 1)).numpy()
    Delta = W_new - W_ref
    return W_new, Delta

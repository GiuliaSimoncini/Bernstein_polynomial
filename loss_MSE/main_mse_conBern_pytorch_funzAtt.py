"""
main_mse_conBern_pytorch_funzAtt.py - self-contained

Ottimizzazione con distanza MSE.
Include l'implementazione dell'attivazione per Ordine Stocastico 
rigoroso (senza uso di penalità) in PyTorch, basata su allocazione residua.

Esperimenti (target = polinomio di Bernstein di grado alto → riduzione a N):
  1-3: Beta pure
  4-6: Polinomi puri
  7-9: Mixture bimodali
  10-12: Convoluzioni numeriche
"""

import os
import time
import numpy as np
import scipy.stats as stats
import scipy.optimize as scopt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import comb

cartella_script = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(cartella_script, 'immagini_mse_conBern_pytorch_funzAtt')
os.makedirs(OUT, exist_ok=True)

# -------
# BASIS
# -------

def basis_matrix(N: int, x: np.ndarray) -> np.ndarray:
    # Matrice base di Bernstein: M[:,k] = C(N,k)*x^k*(1-x)^(N-k)
    M = np.zeros((len(x), N + 1))
    for k in range(N + 1):
        M[:, k] = comb(N, k) * (x ** k) * ((1 - x) ** (N - k))
    return M

def cdf_from_weights(W: np.ndarray, M: np.ndarray, dx: float) -> np.ndarray:
    # Calcola la CDF approssimata dai pesi W
    pdf = np.maximum(M @ W, 0.0)
    cdf = np.cumsum(pdf) * dx
    cdf /= max(cdf[-1], 1e-12)
    return cdf

def bernstein_operator_init(N: int, x: np.ndarray, f: np.ndarray) -> np.ndarray:
    # Inizializzazione con l'operatore di Bernstein classico
    nodes = np.linspace(0, 1, N + 1)
    W = np.interp(nodes, x, f)
    W = np.maximum(W, 0.0)
    total = W.sum()
    return W * (N + 1) / total if total > 1e-12 else np.ones(N + 1)

def mse(W: np.ndarray, M: np.ndarray, f: np.ndarray) -> float:
    # Mean Squared Error (L2)
    return float(np.mean((f - M @ W)**2))

# -----------------
# TARGET FUNCTIONS
# -----------------

def make_bp_target(N_high: int, x: np.ndarray, f_orig: np.ndarray) -> np.ndarray:
    """
    Proietta f_orig su BP di grado N_high (tramite bernstein_operator_init),
    restituisce la PDF risultante normalizzata.
    Utile per creare un target polinomio di grado N_high da ridurre.
    """
    M_high = basis_matrix(N_high, x)
    W_high = bernstein_operator_init(N_high, x, f_orig)
    f_target = M_high @ W_high
    f_target = np.maximum(f_target, 0.0)
    dx = x[1] - x[0]
    norm = np.trapezoid(f_target, x)
    return f_target / max(norm, 1e-12)


def make_bp_convolution_target(N1: int, f1: np.ndarray, N2: int, f2: np.ndarray, x: np.ndarray) -> tuple:
    """
    Esegue la convoluzione discreta dei pesi di due polinomi di Bernstein.
    Restituisce la PDF target (normalizzata) e il grado totale risultante.
    """
    # Calcola i pesi iniziali per le due distribuzioni
    W1 = bernstein_operator_init(N1, x, f1)
    W2 = bernstein_operator_init(N2, x, f2)
    
    # Convoluzione discreta dei pesi (lunghezza: N1 + N2 + 1)
    W_conv = np.convolve(W1, W2, mode='full')
    
    # Il grado risultante è la somma dei gradi
    N_tot = N1 + N2
    
    # Riscala i pesi (la somma in BernOp deve essere N_tot + 1)
    W_conv = W_conv * (N_tot + 1) / (np.sum(W_conv) + 1e-12)
    
    # Proietta sulla matrice di base per ottenere la PDF target
    M_tot = basis_matrix(N_tot, x)
    f_target = M_tot @ W_conv
    f_target = np.maximum(f_target, 0.0)
    
    # Normalizzazione dell'area della PDF
    norm = np.trapezoid(f_target, x)
    f_target = f_target / max(norm, 1e-12)
    
    return f_target, N_tot

# --------------------------------
# STOCHASTIC ORDER & CONSTRAINTS
# --------------------------------

def build_scipy_constraints(N: int, direction) -> list:
    constraints = [{
        'type': 'eq',
        'fun': lambda D: np.sum(D),
        'jac': lambda D: np.ones(N + 1)
    }]
    if direction == 'upper':
        for h in range(N):
            e = np.zeros(N + 1); e[:h + 1] = 1.0
            constraints.append({
                'type': 'ineq',
                'fun': lambda D, h=h: -np.sum(D[:h + 1]),
                'jac': lambda D, e=e.copy(): -e,
            })
    elif direction == 'lower':
        for h in range(1, N + 1):
            e = np.zeros(N + 1); e[h:] = 1.0
            constraints.append({
                'type': 'ineq',
                'fun': lambda D, h=h: -np.sum(D[h:]),
                'jac': lambda D, e=e.copy(): -e,
            })
    return constraints

def check_order(cdf_new: np.ndarray, cdf_target: np.ndarray, direction: str, tol: float = 1e-4) -> dict:
    # Controllo rigoroso sulle vere CDF.
    
    delta_cdf = cdf_new - cdf_target
    if direction == 'upper':
        violations = np.maximum(-delta_cdf, 0.0)
    elif direction == 'lower':
        violations = np.maximum(delta_cdf, 0.0)
    else:
        raise ValueError("direction deve essere 'upper' o 'lower'")
        
    max_viol = float(violations.max())
    return {
        'satisfied': max_viol <= tol,
        'max_violation': max_viol,
    }


# -------------------------
# METODI DI OTTIMIZZAZIONE
# -------------------------

def solve_ing_cane(N: int, x: np.ndarray, f: np.ndarray) -> np.ndarray:
    mode_x = x[np.argmax(f)]
    k = min(int(np.floor(mode_x * N)), N - 1)
    interval = 1.0 / N
    dist_right = (k + 1) / N - mode_x
    dist_left  = mode_x - k / N
    W = np.zeros(N + 1)
    W[k]     = dist_right / interval
    W[k + 1] = dist_left  / interval
    W = W * (N + 1) / W.sum()
    return W

def solve_scipy_mse(N: int, x: np.ndarray, f: np.ndarray,
                   W_init=None, direction=None, W_ref=None):
    M = basis_matrix(N, x)
    n_pts = len(x)

    if W_init is None:
        W_init = bernstein_operator_init(N, x, f)
    if W_ref is None:
        W_ref = W_init.copy()

    Delta_init = W_init - W_ref

    def objective(Delta):
        r = f - M @ (W_ref + Delta)
        return float(np.mean(r**2))

    def jac_obj(Delta):
        r = f - M @ (W_ref + Delta)
        return -2 * (M.T @ r) / n_pts

    constraints = build_scipy_constraints(N, direction)
    bounds = [(-W_ref[k], None) for k in range(N + 1)]

    result = scopt.minimize(
        objective, Delta_init, jac=jac_obj,
        method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 20_000, 'ftol': 1e-13}
    )
    Delta = result.x
    W_new = np.maximum(W_ref + Delta, 0.0)
    Delta = W_new - W_ref
    return W_new, Delta

# MODELLO NON VINCOLATO (PYTORCH)
class _BernsteinModelMSE(nn.Module):
    def __init__(self, N: int):
        super().__init__()
        self.N = N
        self.W_raw = nn.Parameter(torch.zeros(N + 1))

    def forward(self, M_t):
        W = torch.softmax(self.W_raw, dim=0) * (self.N + 1)
        return M_t @ W

    def weights(self):
        with torch.no_grad():
            return (torch.softmax(self.W_raw, dim=0) * (self.N + 1)).numpy()


# MODELLO CON FUNZIONE DI ATTIVAZIONE STOCASTICA (PYTORCH)
class _StochasticOrderedModelMSE(nn.Module):
    """
    Implementa l'algoritmo esatto di allocazione residua della massa descritto 
    nella funzione stochastic_upper_activation. Sostituisce del tutto l'uso di penalità.
    """
    def __init__(self, N: int, W_ref: np.ndarray, direction: str = 'upper'):
        super().__init__()
        self.N = N
        self.direction = direction
        
        # w_ref deve sommare a 1 all'interno dell'algoritmo (vettore di probabilità)
        w_ref_tensor = torch.tensor(W_ref, dtype=torch.float32) / (N + 1)
        self.w_ref = torch.clamp(w_ref_tensor, min=1e-12)
        self.w_ref = self.w_ref / self.w_ref.sum()
        
        # Inizializzazione a 5.0
        # Sigmoid(5) ~= 0.993, che fa allocare fin da subito quasi tutta la 
        # massa disponibile, portando l'output iniziale ad essere W_ref (Delta~=0).
        self.logits = nn.Parameter(torch.ones(N) * 5.0)

    def forward(self, M_t):
        W = self.weights_normalized() * (self.N + 1)
        return M_t @ W

    def weights_normalized(self):
        sigmas = torch.sigmoid(self.logits)
        
        if self.direction == 'upper':
            w_cum = torch.cumsum(self.w_ref, dim=0)
            p = []
            current_sum = torch.tensor(0.0, device=self.logits.device)
            
            for k in range(self.N): # da 0 a N-1
                # Calcola il residuo rispetto al tetto massimo (w_cum[k])
                residuo = torch.clamp(w_cum[k] - current_sum, min=0.0)
                # Assegna una frazione (sigma) della massa residua
                val = residuo * sigmas[k]
                p.append(val)
                current_sum = current_sum + val
            
            # L'ultimo elemento prende tutta la massa rimanente per sommare a 1
            p.append(torch.clamp(1.0 - current_sum, min=0.0))
            return torch.stack(p)
            
        elif self.direction == 'lower':
            # Per il lower bound eseguiamo l'algoritmo al contrario (da destra verso sinistra)
            w_ref_rev = torch.flip(self.w_ref, dims=[0])
            w_cum_rev = torch.cumsum(w_ref_rev, dim=0)
            p_rev = []
            current_sum = torch.tensor(0.0, device=self.logits.device)
            
            for k in range(self.N):
                residuo = torch.clamp(w_cum_rev[k] - current_sum, min=0.0)
                val = residuo * sigmas[k]
                p_rev.append(val)
                current_sum = current_sum + val
            
            p_rev.append(torch.clamp(1.0 - current_sum, min=0.0))
            p_rev_tensor = torch.stack(p_rev)
            
            # Ri-ribaltiamo l'array finale per ottenere l'ordine spaziale corretto
            return torch.flip(p_rev_tensor, dims=[0])

    def weights(self):
        with torch.no_grad():
            return (self.weights_normalized() * (self.N + 1)).numpy()


def solve_pytorch_mse(N: int, x: np.ndarray, f: np.ndarray,
                     epochs: int = 3000, lr: float = 0.05) -> np.ndarray:
    M_t = torch.tensor(basis_matrix(N, x), dtype=torch.float32)
    f_t = torch.tensor(f, dtype=torch.float32)

    model = _BernsteinModelMSE(N)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(M_t), f_t).backward()
        opt.step()

    return model.weights()

def solve_pytorch_ordered_mse(N: int, x: np.ndarray, f: np.ndarray,
                              W_ref: np.ndarray,
                              direction: str = 'upper',
                              epochs: int = 4000,
                              lr: float = 0.05):
    """
    Risolve il problema di ordine stocastico ottimizzando direttamente i logit 
    della nuova funzione di attivazione, senza alcun uso di penalità.
    """
    M_np = basis_matrix(N, x)
    M_t  = torch.tensor(M_np, dtype=torch.float32)
    f_t  = torch.tensor(f,    dtype=torch.float32)

    model = _StochasticOrderedModelMSE(N, W_ref, direction=direction)
    # L'uso della Sigmoide permette di tenere un learning rate vivace
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        opt.zero_grad()
        # Calcola solo l'MSE. I vincoli sono matematicamente garantiti dal forward pass.
        mse_term = loss_fn(model(M_t), f_t)
        mse_term.backward()
        opt.step()

    W_new = model.weights()
    Delta = W_new - W_ref
    return W_new, Delta


# --------
# RUNNER
# --------

METHOD_ORDER = ['bernstein_op', 'bernstein_op_upper', 'bernstein_op_lower',
                'ing_cane', 'scipy', 'scipy_upper', 'scipy_lower',
                'pytorch', 'pytorch_upper', 'pytorch_lower']

def run_experiment(name: str, N: int, x: np.ndarray, f: np.ndarray,
                   epochs_pt: int = 3000,
                   target_label: str = '') -> dict:
    M  = basis_matrix(N, x)
    dx = x[1] - x[0]

    # CDF del target reale
    cdf_t = np.cumsum(np.maximum(f, 0)) * dx
    cdf_t /= max(cdf_t[-1], 1e-12)

    t0 = time.perf_counter()
    W_init = bernstein_operator_init(N, x, f)
    t_bern = (time.perf_counter() - t0) * 1e3

    print(f"\n{'='*70}")
    print(f"  {name}  (N={N})")
    if target_label:
        print(f"  Target: {target_label}")
    print(f"{'='*70}")

    results = dict(name=name, target_label=target_label,
                   N=N, x=x, f=f, M=M, dx=dx)

    def _store(label, W, elapsed, Delta=None):
        m = mse(W, M, f)
        results[label] = dict(W=W, mse=m, time_ms=elapsed, Delta=Delta)
        return m

    # Bernstein operator
    m = _store('bernstein_op', W_init.copy(), t_bern)
    print(f"  Bernstein operator   MSE={m:.7f}   t={t_bern:7.1f} ms")

    # BernOp + SO upper
    t0 = time.perf_counter()
    W_bo_up, D_bo_up = solve_scipy_mse(N, x, f,
                                      W_init=W_init.copy(),
                                      direction='upper',
                                      W_ref=W_init.copy())
    t = (time.perf_counter() - t0) * 1e3
    cdf_a = cdf_from_weights(W_bo_up, M, dx)
    so = check_order(cdf_a, cdf_t, 'upper')
    m = _store('bernstein_op_upper', W_bo_up, t, Delta=D_bo_up)
    results['bernstein_op_upper']['so']    = so
    results['bernstein_op_upper']['W_ref'] = W_init.copy()
    sat = '✓' if so['satisfied'] else '✗'
    print(f"  BernOp+SO upper      MSE={m:.7f}   t={t:7.1f} ms  SO:{sat}")

    # BernOp + SO lower
    t0 = time.perf_counter()
    W_bo_lo, D_bo_lo = solve_scipy_mse(N, x, f,
                                      W_init=W_init.copy(),
                                      direction='lower',
                                      W_ref=W_init.copy())
    t = (time.perf_counter() - t0) * 1e3
    cdf_a = cdf_from_weights(W_bo_lo, M, dx)
    so = check_order(cdf_a, cdf_t, 'lower')
    m = _store('bernstein_op_lower', W_bo_lo, t, Delta=D_bo_lo)
    results['bernstein_op_lower']['so']    = so
    results['bernstein_op_lower']['W_ref'] = W_init.copy()
    sat = '✓' if so['satisfied'] else '✗'
    print(f"  BernOp+SO lower      MSE={m:.7f}   t={t:7.1f} ms  SO:{sat}")

    # Ing-Cane
    t0 = time.perf_counter()
    W_ic = solve_ing_cane(N, x, f)
    t = (time.perf_counter() - t0) * 1e3
    m = _store('ing_cane', W_ic, t)
    print(f"  Ing-Cane             MSE={m:.7f}   t={t:7.1f} ms")

    # Scipy MSE unconstrained
    t0 = time.perf_counter()
    W_scipy, _ = solve_scipy_mse(N, x, f, W_init=W_init.copy())
    t_scipy = (time.perf_counter() - t0) * 1e3
    m = _store('scipy', W_scipy, t_scipy)
    print(f"  Scipy MSE            MSE={m:.7f}   t={t_scipy:7.1f} ms")

    # Scipy + SO upper
    t0 = time.perf_counter()
    W_up, D_up = solve_scipy_mse(N, x, f,
                                 W_init=W_scipy.copy(),
                                 direction='upper',
                                 W_ref=W_init.copy())
    t = (time.perf_counter() - t0) * 1e3
    cdf_a = cdf_from_weights(W_up, M, dx)
    so_up = check_order(cdf_a, cdf_t, 'upper')
    m = _store('scipy_upper', W_up, t, Delta=D_up)
    results['scipy_upper']['so']    = so_up
    results['scipy_upper']['W_ref'] = W_init.copy()
    sat = '✓' if so_up['satisfied'] else '✗'
    print(f"  Scipy+SO upper       MSE={m:.7f}   t={t:7.1f} ms  SO:{sat}")

    # Scipy + SO lower
    t0 = time.perf_counter()
    W_lo, D_lo = solve_scipy_mse(N, x, f,
                                 W_init=W_scipy.copy(),
                                 direction='lower',
                                 W_ref=W_init.copy())
    t = (time.perf_counter() - t0) * 1e3
    cdf_a = cdf_from_weights(W_lo, M, dx)
    so_lo = check_order(cdf_a, cdf_t, 'lower')
    m = _store('scipy_lower', W_lo, t, Delta=D_lo)
    results['scipy_lower']['so']    = so_lo
    results['scipy_lower']['W_ref'] = W_init.copy()
    sat = '✓' if so_lo['satisfied'] else '✗'
    print(f"  Scipy+SO lower       MSE={m:.7f}   t={t:7.1f} ms  SO:{sat}")

    # PyTorch MSE unconstrained
    t0 = time.perf_counter()
    W_pt = solve_pytorch_mse(N, x, f, epochs=epochs_pt)
    t_pytorch = (time.perf_counter() - t0) * 1e3
    m = _store('pytorch', W_pt, t_pytorch)
    print(f"  PyTorch MSE          MSE={m:.7f}   t={t_pytorch:7.1f} ms")

    # PyTorch + SO upper (Nuova Attivazione)
    t0 = time.perf_counter()
    W_pt_up, D_pt_up = solve_pytorch_ordered_mse(N, x, f, W_init.copy(),
                                                 direction='upper', epochs=epochs_pt)
    t = (time.perf_counter() - t0) * 1e3
    cdf_a = cdf_from_weights(W_pt_up, M, dx)
    so = check_order(cdf_a, cdf_t, 'upper')
    m = _store('pytorch_upper', W_pt_up, t, Delta=D_pt_up)
    results['pytorch_upper']['so']    = so
    results['pytorch_upper']['W_ref'] = W_init.copy()
    sat = '✓' if so['satisfied'] else '✗'
    print(f"  PyTorch+SO (Attiv.) upp  MSE={m:.7f}   t={t:7.1f} ms  SO:{sat}")

    # PyTorch + SO lower (Nuova Attivazione)
    t0 = time.perf_counter()
    W_pt_up, D_pt_up = solve_pytorch_ordered_mse(N, x, f, W_init.copy(),
                                                 direction='lower', epochs=epochs_pt)
    t = (time.perf_counter() - t0) * 1e3
    cdf_a = cdf_from_weights(W_pt_up, M, dx)
    so = check_order(cdf_a, cdf_t, 'lower')
    m = _store('pytorch_lower', W_pt_up, t, Delta=D_pt_up)
    results['pytorch_lower']['so']    = so
    results['pytorch_lower']['W_ref'] = W_init.copy()
    sat = '✓' if so['satisfied'] else '✗'
    print(f"  PyTorch+SO (Attiv.) low  MSE={m:.7f}   t={t:7.1f} ms  SO:{sat}")

    # Riepilogo
    print(f"\n  {'Metodo':<28}  {'MSE':>12}  {'Tempo (ms)':>12}  SO")
    print(f"  {'-'*58}")
    for k in METHOD_ORDER:
        if k not in results:
            continue
        mv = results[k]['mse']
        tv = results[k]['time_ms']
        so_info = results[k].get('so', {})
        sat = f"  {'✓' if so_info.get('satisfied', False) else '✗'}" if so_info else ''
        print(f"  {k:<28}  {mv:12.7f}  {tv:12.1f}{sat}")

    return results


# ----------
# PALETTE
# ----------

STYLE = {
    'bernstein_op':       dict(color='#95a5a6', ls='--', lw=1.6, label='Bernstein operator'),
    'bernstein_op_upper': dict(color='#f39c12', ls=':',  lw=1.8, label='BernOp + SO upper'),
    'bernstein_op_lower': dict(color='#1abc9c', ls=':',  lw=1.8, label='BernOp + SO lower'),
    'ing_cane':           dict(color='#e74c3c', ls='-',  lw=1.8, label='Ing-Cane'),
    'scipy':              dict(color='#27ae60', ls='-',  lw=2.0, label='Scipy MSE SLSQP'),
    'scipy_upper':        dict(color='#e67e22', ls='-',  lw=1.8, label='Scipy MSE + SO upper'),
    'scipy_lower':        dict(color='#8e44ad', ls='-',  lw=1.8, label='Scipy MSE + SO lower'),
    'pytorch':            dict(color='#2980b9', ls='--', lw=1.8, label='PyTorch MSE'),
    'pytorch_upper':      dict(color='#c0392b', ls='--', lw=1.6, label='PyTorch MSE + SO upp (Attivaz.)'),
    'pytorch_lower':      dict(color="#c02b93", ls='--', lw=1.6, label='PyTorch MSE + SO low (Attivaz.)')
}

def _cdf_target(f: np.ndarray, dx: float) -> np.ndarray:
    cdf = np.cumsum(np.maximum(f, 0)) * dx
    return cdf / max(cdf[-1], 1e-12)

def _full_title(results: dict, subtitle: str = '') -> str:
    tl = results.get('target_label', '')
    base = f"{results['name']}  (N={results['N']})"
    if tl:
        base += f"\nTarget: {tl}"
    if subtitle:
        base += f"  |  {subtitle}"
    return base


# -------------------
# PLOTTING FUNCTIONS
# -------------------

def plot_pdf_comparison(results: dict, save_path=None):
    x, f, M = results['x'], results['f'], results['M']
    N = results['N']

    fig, (ax_pdf, ax_mse) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(_full_title(results, 'Confronto PDF'), fontsize=11, fontweight='bold')

    ax_pdf.plot(x, f, 'k--', lw=2.5, zorder=6, label='f(x) target')
    for key, st in STYLE.items():
        if key not in results:
            continue
        ax_pdf.plot(x, M @ results[key]['W'], **st, alpha=0.85)
    ax_pdf.set_xlabel('x'); ax_pdf.set_ylabel('Densità'); ax_pdf.set_title('PDF')
    ax_pdf.legend(fontsize=7); ax_pdf.grid(True, alpha=0.3)

    keys = [k for k in STYLE if k in results]
    mses  = [results[k]['mse']      for k in keys]
    times = [results[k]['time_ms']  for k in keys]
    cols  = [STYLE[k]['color']      for k in keys]
    lbls  = [STYLE[k]['label']      for k in keys]

    bars = ax_mse.bar(range(len(keys)), mses, color=cols, alpha=0.8)
    ax_mse.set_xticks(range(len(keys)))
    ax_mse.set_xticklabels(lbls, rotation=40, ha='right', fontsize=7)
    ax_mse.set_ylabel('MSE (L2)'); ax_mse.set_title('MSE per metodo')
    ax_mse.grid(True, axis='y', alpha=0.3)
    for bar, t in zip(bars, times):
        h = bar.get_height()
        label = f'{t:.0f}ms' if t > 0.5 else '<1ms'
        ax_mse.text(bar.get_x() + bar.get_width() / 2, h * 1.04,
                    label, ha='center', va='bottom', fontsize=6.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_cdf_all(results: dict, save_path=None):
    x, f, M, dx = results['x'], results['f'], results['M'], results['dx']
    cdf_t = _cdf_target(f, dx)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(_full_title(results, 'CDF tutti i metodi'), fontsize=11, fontweight='bold')
    ax.plot(x, cdf_t, 'k--', lw=2.5, zorder=6, label='CDF target')
    for key, st in STYLE.items():
        if key not in results:
            continue
        cdf_a = cdf_from_weights(results[key]['W'], M, dx)
        ax.plot(x, cdf_a, color=st['color'], ls=st['ls'],
                lw=st['lw'], label=st['label'], alpha=0.85)
    ax.set_xlabel('x'); ax.set_ylabel('CDF')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_weights(results: dict, save_path=None):
    N = results['N']
    nodes = np.arange(N + 1)
    keys = [k for k in ['bernstein_op', 'scipy', 'scipy_upper', 'scipy_lower',
                         'bernstein_op_upper', 'bernstein_op_lower', 'pytorch', 'pytorch_upper', 'pytorch_lower']
            if k in results]

    fig, axes = plt.subplots(1, len(keys), figsize=(len(keys) * 3.2, 4), sharey=True)
    if len(keys) == 1:
        axes = [axes]
    fig.suptitle(_full_title(results, 'Pesi W'), fontsize=10, fontweight='bold')

    for ax, key in zip(axes, keys):
        st = STYLE[key]
        W  = results[key]['W']
        ax.bar(nodes, W, color=st['color'], alpha=0.75)
        ax.set_title(f'{st["label"]}\nMSE={results[key]["mse"]:.5f}', fontsize=7)
        ax.set_xlabel('k'); ax.grid(True, axis='y', alpha=0.3)
    axes[0].set_ylabel('W[k]')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_delta(results: dict, save_path=None):
    N = results['N']
    nodes = np.arange(N + 1)
    so_keys = [k for k in ['bernstein_op_upper', 'bernstein_op_lower',
                            'scipy_lower', 'scipy_upper', 'pytorch_upper', 'pytorch_lower']
               if k in results and results[k].get('Delta') is not None]
    if not so_keys:
        return

    fig, axes = plt.subplots(2, len(so_keys), figsize=(len(so_keys) * 3.5, 7))
    if len(so_keys) == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(_full_title(results, 'Delta = W_new - W_ref'), fontsize=10, fontweight='bold')

    for j, key in enumerate(so_keys):
        st = STYLE[key]
        Delta = results[key]['Delta']
        so = results[key].get('so', {})
        sat = '✓' if so.get('satisfied', False) else '✗'

        ax_d = axes[0, j]
        colors = [st['color'] if d > 0 else '#c0392b' for d in Delta]
        ax_d.bar(nodes, Delta, color=colors, alpha=0.75)
        ax_d.axhline(0, color='black', lw=0.8)
        ax_d.set_title(f'{st["label"]}\n'
                       f'|Δ|max={np.abs(Delta).max():.3f}  sum={Delta.sum():.1e}  SO:{sat}',
                       fontsize=7.5)
        ax_d.set_xlabel('k'); ax_d.set_ylabel('Δ[k]'); ax_d.grid(True, axis='y', alpha=0.3)

        ax_c = axes[1, j]
        direction = 'upper' if 'upper' in key else 'lower'
        if direction == 'upper':
            cs = np.cumsum(Delta)
            thresh_label = 'cumsum l→r  [≤0 richiesto]'
        else:
            cs = np.cumsum(Delta[::-1])[::-1]
            thresh_label = 'cumsum r→l  [≤0 richiesto]'
        ax_c.bar(nodes, cs, color=st['color'], alpha=0.55)
        ax_c.axhline(0, color='black', lw=1.0)
        ax_c.set_title(thresh_label, fontsize=7.5)
        ax_c.set_xlabel('k'); ax_c.set_ylabel('tail-sum Δ'); ax_c.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_so_single(results: dict, key: str, direction: str, save_path=None):
    x, f, M, dx = results['x'], results['f'], results['M'], results['dx']
    W  = results[key]['W']
    st = STYLE[key]
    so = results[key].get('so', {})

    cdf_t = _cdf_target(f, dx)
    cdf_a = cdf_from_weights(W, M, dx)
    delta_cdf = cdf_a - cdf_t
    idx_crit  = np.argmin(np.abs(delta_cdf))
    x_crit = x[idx_crit]
    zoom_hw = 0.15
    x_lo = max(x[0], x_crit - zoom_hw)
    x_hi = min(x[-1], x_crit + zoom_hw)
    mask = (x >= x_lo) & (x <= x_hi)

    is_satisfied = so.get('satisfied', False)
    sat_str = '✓ Soddisfatto' if is_satisfied else '✗ Violato'
    tick_sym = '✓' if is_satisfied else '✗'

    dir_str = 'upper (F_approx ≥ F_target)' if direction == 'upper' \
              else 'lower (F_approx ≤ F_target)'

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        _full_title(results,
                    f'SO {dir_str} — {st["label"]} — {sat_str}\n'
                    f'margine min={np.abs(delta_cdf).min():.4f}'),
        fontsize=9, fontweight='bold')

    for ax, xd, ct, ca, title in [
        (ax_full, x,       cdf_t,       cdf_a,       'Vista completa'),
        (ax_zoom, x[mask], cdf_t[mask], cdf_a[mask],
         f'Zoom x∈[{x_lo:.2f},{x_hi:.2f}]'),
    ]:
        ax.plot(xd, ct, 'k--', lw=2.2, label='CDF target')
        ax.plot(xd, ca, color=st['color'], lw=2, label=f'CDF {st["label"]}')

        if direction == 'upper':
            ax.fill_between(xd, ct, ca, where=(ca >= ct),
                            color=st['color'], alpha=0.18, 
                            label=f'F_approx ≥ F_target {tick_sym}')
        else:
            ax.fill_between(xd, ca, ct, where=(ca <= ct),
                            color=st['color'], alpha=0.18, 
                            label=f'F_approx ≤ F_target {tick_sym}')
                            
        ax.set_xlabel('x'); ax.set_ylabel('CDF'); ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax_full.axvspan(x_lo, x_hi, color='gray', alpha=0.10, label='zona zoom')
    ax_full.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_so_summary(results: dict, save_path=None):
    x, f, M, dx = results['x'], results['f'], results['M'], results['dx']
    cdf_t = _cdf_target(f, dx)
    so_keys = [k for k in ['bernstein_op', 'bernstein_op_upper', 'bernstein_op_lower',
                            'scipy', 'scipy_upper', 'scipy_lower',
                            'pytorch', 'pytorch_upper', 'pytorch_lower'] if k in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(_full_title(results, 'Riepilogo Ordine Stocastico (MSE)'),
                 fontsize=11, fontweight='bold')
    ax.plot(x, cdf_t, 'k--', lw=2.5, zorder=6, label='CDF target')

    if 'scipy_upper' in results and 'scipy_lower' in results:
        cdf_up = cdf_from_weights(results['scipy_upper']['W'], M, dx)
        cdf_lo = cdf_from_weights(results['scipy_lower']['W'], M, dx)
        ax.fill_between(x, cdf_lo, cdf_up, color='#27ae60', alpha=0.08,
                        label='corridoio SO Scipy', zorder=1)
    if 'bernstein_op_upper' in results and 'bernstein_op_lower' in results:
        cdf_bo_up = cdf_from_weights(results['bernstein_op_upper']['W'], M, dx)
        cdf_bo_lo = cdf_from_weights(results['bernstein_op_lower']['W'], M, dx)
        ax.fill_between(x, cdf_bo_lo, cdf_bo_up, color='#f39c12', alpha=0.07,
                        label='corridoio SO BernOp', zorder=1)

    for key in so_keys:
        st = STYLE[key]
        cdf_a = cdf_from_weights(results[key]['W'], M, dx)
        ax.plot(x, cdf_a, color=st['color'], ls=st['ls'],
                lw=st['lw'], label=st['label'], alpha=0.88, zorder=3)

    ax.set_xlabel('x'); ax.set_ylabel('CDF')
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# -----------------------------------------------
# HELPER: salva tutti i plot per un esperimento
# -----------------------------------------------

def save_all_plots(r: dict, prefix: str):
    plot_pdf_comparison(r, save_path=f'{OUT}/{prefix}_pdf.png')
    plot_cdf_all(r, save_path=f'{OUT}/{prefix}_cdf.png')
    plot_weights(r, save_path=f'{OUT}/{prefix}_weights.png')
    plot_delta(r, save_path=f'{OUT}/{prefix}_delta.png')
    plot_so_single(r, 'bernstein_op_upper', 'upper',
                   save_path=f'{OUT}/{prefix}_bo_so_upper.png')
    plot_so_single(r, 'bernstein_op_lower', 'lower',
                   save_path=f'{OUT}/{prefix}_bo_so_lower.png')
    plot_so_single(r, 'scipy_upper', 'upper',
                   save_path=f'{OUT}/{prefix}_so_upper.png')
    plot_so_single(r, 'scipy_lower', 'lower',
                   save_path=f'{OUT}/{prefix}_so_lower.png')
    plot_so_summary(r, save_path=f'{OUT}/{prefix}_so_summary.png')


# -----
# MAIN
# -----
if __name__ == '__main__':

    x = np.linspace(0.001, 0.999, 400)

    # ---------------------------------------
    # GRUPPO A: TARGET BETA (mappati su BP)
    # ---------------------------------------

    # ---------------------------------------------
    # Esp 1 - Beta grado 65 -> Approssimante N=32
    # ---------------------------------------------
    print("\n Esperimento 1: Target BP da Beta(33,34) (grado 65) → N=32")
    f1_orig = stats.beta(33, 34).pdf(x)
    f1 = make_bp_target(65, x, f1_orig)
    r1 = run_experiment(
        name='Target BP Beta (grado 65)',
        N=32, x=x, f=f1, epochs_pt=3000,
        target_label='BP grado 65 da Beta(33,34) → N=32'
    )
    save_all_plots(r1, 'exp1')

    # --------------------------------------------
    # Esp 2 - Beta grado 33 -> Approssimante N=16
    # ---------------------------------------------
    print("\n Esperimento 2: Target BP da Beta(17,18) (grado 33) → N=16")
    f2_orig = stats.beta(17, 18).pdf(x)
    f2 = make_bp_target(33, x, f2_orig)
    r2 = run_experiment(
        name='Target BP Beta (grado 33)',
        N=16, x=x, f=f2, epochs_pt=3000,
        target_label='BP grado 33 da Beta(17,18) → N=16'
    )
    save_all_plots(r2, 'exp2')

    # --------------------------------------------
    # Esp 3 - Beta grado 17 -> Approssimante N=8
    # ---------------------------------------------
    print("\n Esperimento 3: Target BP da Beta(9,10) (grado 17) → N=8")
    f3_orig = stats.beta(9, 10).pdf(x)
    f3 = make_bp_target(17, x, f3_orig)
    r3 = run_experiment(
        name='Target BP Beta (grado 17)',
        N=8, x=x, f=f3, epochs_pt=3000,
        target_label='BP grado 17 da Beta(9,10) → N=8'
    )
    save_all_plots(r3, 'exp3')


    # -------------------------------------------------
    # GRUPPO B: TARGET POLINOMIO (mappati su BP)
    # -------------------------------------------------

    # --------------------------------------------------
    # Esp 4 - Polinomio grado 65 -> Approssimante N=32
    # --------------------------------------------------
    print("\n Esperimento 4: Target BP da Polinomio x³³(1-x)³² (grado 65) → N=32")
    f4_orig = (x ** 33) * ((1 - x) ** 32)
    f4_orig = f4_orig / np.max(f4_orig)
    f4 = make_bp_target(65, x, f4_orig)
    r4 = run_experiment(
        name='BP Polinomio puro (grado 65)',
        N=32, x=x, f=f4, epochs_pt=3000,
        target_label='BP grado 65 da Polinomio x³³(1-x)³² → N=32'
    )
    save_all_plots(r4, 'exp4')

    # -------------------------------------------------
    # Esp 5 - Polinomio grado 33 -> Approssimante N=16
    # -------------------------------------------------
    print("\n Esperimento 5: Target BP da Polinomio x¹⁷(1-x)¹⁶ (grado 33) → N=16")
    f5_orig = (x ** 17) * ((1 - x) ** 16)
    f5_orig = f5_orig / np.max(f5_orig)
    f5 = make_bp_target(33, x, f5_orig)
    r5 = run_experiment(
        name='BP Polinomio puro (grado 33)',
        N=16, x=x, f=f5, epochs_pt=3000,
        target_label='BP grado 33 da Polinomio x¹⁷(1-x)¹⁶ → N=16'
    )
    save_all_plots(r5, 'exp5')

    # -------------------------------------------------
    # Esp 6 - Polinomio grado 17 -> Approssimante N=8
    # -------------------------------------------------
    print("\n Esperimento 6: Target BP da Polinomio x⁹(1-x)⁸ (grado 17) → N=8")
    f6_orig = (x ** 9) * ((1 - x) ** 8)
    f6_orig = f6_orig / np.max(f6_orig)
    f6 = make_bp_target(17, x, f6_orig)
    r6 = run_experiment(
        name='BP Polinomio puro (grado 17)',
        N=8, x=x, f=f6, epochs_pt=3000,
        target_label='BP grado 17 da Polinomio x⁹(1-x)⁸ → N=8'
    )
    save_all_plots(r6, 'exp6')


    # --------------------------------------------------
    # GRUPPO C: TARGET MIXTURE BIMODALE (mappati su BP)
    # ---------------------------------------------------

    # -------------------------------------------------
    # Esp 7 - Mixture grado 65 -> Approssimante N=32
    # -------------------------------------------------
    print("\n Esperimento 7: Target BP da Beta mixture bimodale (grado 65) → N=32")
    f7_orig = 0.5 * stats.beta(15, 52).pdf(x) + 0.5 * stats.beta(52, 15).pdf(x)
    f7 = make_bp_target(65, x, f7_orig)
    r7 = run_experiment(
        name='BP Beta Mixture bimodale (grado 65)',
        N=32, x=x, f=f7, epochs_pt=4000,
        target_label='BP grado 65 da Mix 0.5·Beta(15,52)+0.5·Beta(52,15) → N=32'
    )
    save_all_plots(r7, 'exp7')

    # -----------------------------------------------
    # Esp 8 - Mixture grado 33 -> Approssimante N=16
    # -----------------------------------------------
    print("\n Esperimento 8: Target BP da Beta mixture bimodale (grado 33) → N=16")
    f8_orig = 0.5 * stats.beta(8, 27).pdf(x) + 0.5 * stats.beta(27, 8).pdf(x)
    f8 = make_bp_target(33, x, f8_orig)
    r8 = run_experiment(
        name='BP Beta Mixture bimodale (grado 33)',
        N=16, x=x, f=f8, epochs_pt=4000,
        target_label='BP grado 33 da Mix 0.5·Beta(8,27)+0.5·Beta(27,8) → N=16'
    )
    save_all_plots(r8, 'exp8')

    # ------------------------------------------------
    # Esp 9 - Mixture grado 17 -> Approssimante N=8
    # ------------------------------------------------
    print("\n Esperimento 9: Target BP da Beta mixture bimodale (grado 17) → N=8")
    f9_orig = 0.5 * stats.beta(4, 15).pdf(x) + 0.5 * stats.beta(15, 4).pdf(x)
    f9 = make_bp_target(17, x, f9_orig)
    r9 = run_experiment(
        name='BP Beta Mixture bimodale (grado 17)',
        N=8, x=x, f=f9, epochs_pt=4000,
        target_label='BP grado 17 da Mix 0.5·Beta(4,15)+0.5·Beta(15,4) → N=8'
    )
    save_all_plots(r9, 'exp9')


    # ------------------------------
    # GRUPPO D: TARGET CONVOLUZIONE
    # ------------------------------

    # -------------------------------------------------------
    # Esp 10 - Convoluzione grado 65 -> Approssimante N=32
    # -------------------------------------------------------
    print("\n Esperimento 10: Convoluzione BP (grado 32 + 33 = 65) → N=32")
    f_c10_a = stats.beta(12, 22).pdf(x)
    f_c10_b = stats.beta(22, 13).pdf(x)
    f10_conv, N10_tot = make_bp_convolution_target(32, f_c10_a, 33, f_c10_b, x)
    
    r10 = run_experiment(
        name='Convoluzione di BP (grado 65)',
        N=32, x=x, f=f10_conv, epochs_pt=4000,
        target_label=f'BP convoluzione {N10_tot} → N=32'
    )
    save_all_plots(r10, 'exp10')

    # -------------------------------------------------------
    # Esp 11 - Convoluzione grado 33 -> Approssimante N=16
    # -------------------------------------------------------
    print("\n Esperimento 11: Convoluzione BP (grado 16 + 17 = 33) → N=16")
    f_c11_a = stats.beta(5, 13).pdf(x)
    f_c11_b = stats.beta(13, 6).pdf(x)
    f11_conv, N11_tot = make_bp_convolution_target(16, f_c11_a, 17, f_c11_b, x)
    
    r11 = run_experiment(
        name='Convoluzione di BP (grado 33)',
        N=16, x=x, f=f11_conv, epochs_pt=4000,
        target_label=f'BP convoluzione {N11_tot} → N=16'
    )
    save_all_plots(r11, 'exp11')

    # -------------------------------------------------------
    # Esp 12 - Convoluzione grado 17 -> Approssimante N=8
    # -------------------------------------------------------
    print("\n Esperimento 12: Convoluzione BP (grado 8 + 9 = 17) → N=8")
    f_c12_a = stats.beta(3, 7).pdf(x)
    f_c12_b = stats.beta(7, 4).pdf(x)
    f12_conv, N12_tot = make_bp_convolution_target(8, f_c12_a, 9, f_c12_b, x)
    
    r12 = run_experiment(
        name='Convoluzione di BP (grado 17)',
        N=8, x=x, f=f12_conv, epochs_pt=4000,
        target_label=f'BP convoluzione {N12_tot} → N=8'
    )
    save_all_plots(r12, 'exp12')

    print(f"\n{'='*140}")
    print(f"  Tutti i 12 esperimenti sono stati completati.")
    print(f"  Output salvati in: {OUT}/")
    print(f"{'='*140}")
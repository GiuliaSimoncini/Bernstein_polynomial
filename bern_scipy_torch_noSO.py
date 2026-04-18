"""
bern_scipy_torch_noSO.py

Versione ridotta: solo Bernstein operator, Scipy SLSQP e PyTorch,
nessun vincolo di ordinamento stocastico.

Per ogni esperimento produce:
  expN_simple_pdf.png     : confronto PDF + barplot MSE
  expN_simple_cdf.png     : CDF dei tre metodi vs CDF target
  expN_simple_weights.png : istogramma pesi W
"""

import os
import numpy as np
import scipy.stats as stats
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bernstein.basis import basis_matrix, bernstein_operator_init, mse, cdf_from_weights
from bernstein.methods import solve_scipy, solve_pytorch

OUT = 'immagini_bern_scipy_torch_noSO'
os.makedirs(OUT, exist_ok=True)

N = 60
x = np.linspace(0.001, 0.999, 400)

# Palette (sottoinsieme del plotting.py originale)

STYLE = {
    'bernstein_op': dict(color='#95a5a6', ls='--', lw=1.8, label='Bernstein operator'),
    'scipy': dict(color='#27ae60', ls='-',  lw=2.0, label='Scipy SLSQP'),
    'pytorch': dict(color='#2980b9', ls='--', lw=1.8, label='PyTorch'),
}


# Runner minimo

def run_simple(name: str, N: int, x: np.ndarray, f: np.ndarray,
               epochs_pt: int = 3000) -> dict:
    """
    Esegue i tre metodi senza SO e restituisce un dict con W, MSE e tempo
    per ciascuno, più i metadati necessari al plotting.
    """
    M  = basis_matrix(N, x)
    dx = x[1] - x[0]

    t0 = time.perf_counter()
    W_init = bernstein_operator_init(N, x, f)
    t = (time.perf_counter() - t0) * 1e3

    print(f"\n{'='*60}")
    print(f"  {name}  (N={N})")
    print(f"{'='*60}")

    results = dict(name=name, N=N, x=x, f=f, M=M, dx=dx)

    # 1. Bernstein operator (nessuna ottimizzazione)
    m = mse(W_init, M, f)
    results['bernstein_op'] = dict(W=W_init.copy(), mse=m, time_ms=t)
    print(f"  Bernstein operator   MSE={m:.7f}   t={t:8.2f} ms")

    # 2. Scipy SLSQP (non vincolato - solo positività e sum = N+1)
    t0 = time.perf_counter()
    W_scipy, _ = solve_scipy(N, x, f, W_init=W_init.copy())
    t = (time.perf_counter() - t0) * 1e3
    m = mse(W_scipy, M, f)
    results['scipy'] = dict(W=W_scipy, mse=m, time_ms=t)
    print(f"  Scipy SLSQP          MSE={m:.7f}   t={t:8.2f} ms")

    # 3. PyTorch gradient descent (non vincolato)
    t0 = time.perf_counter()
    W_pt = solve_pytorch(N, x, f, epochs=epochs_pt)
    t = (time.perf_counter() - t0) * 1e3
    m = mse(W_pt, M, f)
    results['pytorch'] = dict(W=W_pt, mse=m, time_ms=t)
    print(f"  PyTorch              MSE={m:.7f}   t={t:8.2f} ms")

    # Riepilogo
    print(f"\n  {'Metodo':<22}  {'MSE':>12}  {'Tempo (ms)':>12}")
    print(f"  {'-'*48}")
    for k in ['bernstein_op', 'scipy', 'pytorch']:
        print(f"  {k:<22}  {results[k]['mse']:12.7f}  {results[k]['time_ms']:12.1f}")

    return results


# Plotting

def _cdf_target(f: np.ndarray, dx: float) -> np.ndarray:
    cdf = np.cumsum(np.maximum(f, 0)) * dx
    return cdf / max(cdf[-1], 1e-12)


def plot_simple_pdf(results: dict, save_path: str | None = None):
    """PDF dei tre metodi vs f(x) + barplot MSE con tempi."""
    x, f, M = results['x'], results['f'], results['M']
    N, name  = results['N'], results['name']

    fig, (ax_pdf, ax_mse) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Approssimazione PDF — {name}  (N={N})',
                 fontsize=13, fontweight='bold')

    ax_pdf.plot(x, f, 'k--', lw=2.5, zorder=6, label='f(x) target')
    for key, st in STYLE.items():
        ax_pdf.plot(x, M @ results[key]['W'], **st, alpha=0.85)

    ax_pdf.set_xlabel('x')
    ax_pdf.set_ylabel('Densità')
    ax_pdf.set_title('PDF')
    ax_pdf.legend(fontsize=8)
    ax_pdf.grid(True, alpha=0.3)

    keys  = list(STYLE.keys())
    mses  = [results[k]['mse']     for k in keys]
    times = [results[k]['time_ms'] for k in keys]
    cols  = [STYLE[k]['color']     for k in keys]
    lbls  = [STYLE[k]['label']     for k in keys]

    bars = ax_mse.bar(range(len(keys)), mses, color=cols, alpha=0.8)
    ax_mse.set_xticks(range(len(keys)))
    ax_mse.set_xticklabels(lbls, rotation=30, ha='right', fontsize=8)
    ax_mse.set_ylabel('MSE')
    ax_mse.set_title('MSE per metodo')
    ax_mse.grid(True, axis='y', alpha=0.3)
    for bar, t in zip(bars, times):
        h     = bar.get_height()
        label = f'{t:.0f}ms' if t > 0.5 else '<1ms'
        ax_mse.text(bar.get_x() + bar.get_width() / 2, h * 1.04,
                    label, ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_simple_cdf(results: dict, save_path: str | None = None):
    """CDF numerica dei tre metodi vs CDF target."""
    x, f, M, dx = results['x'], results['f'], results['M'], results['dx']
    N, name      = results['N'], results['name']
    cdf_t        = _cdf_target(f, dx)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f'CDF — {name}  (N={N})', fontsize=12, fontweight='bold')

    ax.plot(x, cdf_t, 'k--', lw=2.5, zorder=6, label='CDF target')
    for key, st in STYLE.items():
        cdf_a = cdf_from_weights(results[key]['W'], M, dx)
        ax.plot(x, cdf_a, color=st['color'], ls=st['ls'],
                lw=st['lw'], label=st['label'], alpha=0.85)

    ax.set_xlabel('x')
    ax.set_ylabel('CDF')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_simple_weights(results: dict, save_path: str | None = None):
    """Istogramma pesi W per i tre metodi."""
    N     = results['N']
    nodes = np.arange(N + 1)
    keys  = list(STYLE.keys())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    fig.suptitle(f'Pesi W — {results["name"]}  (N={N})',
                 fontsize=11, fontweight='bold')

    for ax, key in zip(axes, keys):
        st = STYLE[key]
        W  = results[key]['W']
        ax.bar(nodes, W, color=st['color'], alpha=0.75)
        ax.set_title(f'{st["label"]}\nMSE={results[key]["mse"]:.5f}',
                     fontsize=8)
        ax.set_xlabel('k')
        ax.grid(True, axis='y', alpha=0.3)

    axes[0].set_ylabel('W[k]')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# --------------------------------------------------
# Esperimento 1 - Beta(4,10): unimodale, moda ~0.27
# --------------------------------------------------

f1 = stats.beta(4, 10).pdf(x)

r1 = run_simple('Beta(4,10) — unimodale', N, x, f1, epochs_pt=3000)
plot_simple_pdf(r1, save_path=f'{OUT}/exp1_simple_pdf.png')
plot_simple_cdf(r1, save_path=f'{OUT}/exp1_simple_cdf.png')
plot_simple_weights(r1, save_path=f'{OUT}/exp1_simple_weights.png')
print("  → Esperimento 1 completato.")

# ----------------------------------------------------
# Esperimento 2 - Polinomio x^3(1-x)^2, moda in x=0.6
# ----------------------------------------------------
f2_raw = x ** 3 * (1 - x) ** 2
f2     = f2_raw / np.trapezoid(f2_raw, x)

r2 = run_simple('Polinomio x³(1-x)²  (moda=0.6)', N, x, f2, epochs_pt=5000)
plot_simple_pdf(r2, save_path=f'{OUT}/exp2_simple_pdf.png')
plot_simple_cdf(r2, save_path=f'{OUT}/exp2_simple_cdf.png')
plot_simple_weights(r2, save_path=f'{OUT}/exp2_simple_weights.png')
print("  → Esperimento 2 completato.")

# ----------------------------------------------------
# Esperimento 3 - Beta mixture bimodale (bonus)
# 0.5*Beta(3,10) + 0.5*Beta(10,3), moda ~0.18 e ~0.82
# ----------------------------------------------------
f3 = 0.5 * stats.beta(3, 10).pdf(x) + 0.5 * stats.beta(10, 3).pdf(x)

r3 = run_simple('Beta mixture bimodale', N, x, f3, epochs_pt=5000)
plot_simple_pdf(r3, save_path=f'{OUT}/exp3_simple_pdf.png')
plot_simple_cdf(r3, save_path=f'{OUT}/exp3_simple_cdf.png')
plot_simple_weights(r3, save_path=f'{OUT}/exp3_simple_weights.png')
print("  → Esperimento 3 (bimodale) completato.")

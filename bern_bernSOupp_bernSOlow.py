"""
bern_bernSOupp_bernSOlow.py

Solo metodi basati sull'operatore di Bernstein:

  bernstein_op       : operatore classico, nessuna ottimizzazione, W[k] = f(k/N)
  bernstein_op_upper : SO upper rispetto a W_bernstein_op  (SLSQP vincolato)
  bernstein_op_lower : SO lower rispetto a W_bernstein_op  (SLSQP vincolato)

Per ogni esperimento produce:
  expN_bern_pdf.png     : confronto PDF + barplot MSE
  expN_bern_cdf.png     : CDF dei tre metodi vs CDF target
  expN_bern_weights.png : istogramma pesi W
  expN_bern_delta.png   : istogramma Delta + cumsum per upper e lower
  expN_bern_so.png      : riepilogo SO con corridoio upper/lower
"""

import os
import time
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bernstein.basis import (basis_matrix, bernstein_operator_init,
                             mse, cdf_from_weights)
from bernstein.methods import solve_scipy
from bernstein.stochastic import check_order

OUT = 'immagini_bern_bernSOupp_bernSOlow'
os.makedirs(OUT, exist_ok=True)

N = 100
x = np.linspace(0.001, 0.999, 400)

# Palette 

STYLE = {
    'bernstein_op':       dict(color='#95a5a6', ls='--', lw=2.0,
                               label='Bernstein operator'),
    'bernstein_op_upper': dict(color='#e67e22', ls='-',  lw=2.0,
                               label='BernOp + SO upper'),
    'bernstein_op_lower': dict(color='#8e44ad', ls='-',  lw=2.0,
                               label='BernOp + SO lower'),
}


# Runner

def run_bernstein(name: str, N: int, x: np.ndarray, f: np.ndarray) -> dict:
    """
    Esegue i tre metodi Bernstein e restituisce un dict con W, MSE,
    tempo, Delta e info SO per ciascuno.
    """
    M  = basis_matrix(N, x)
    dx = x[1] - x[0]

    t0 = time.perf_counter()
    W_init = bernstein_operator_init(N, x, f)
    t_bern = (time.perf_counter() - t0) * 1e3

    print(f"\n{'='*60}")
    print(f"  {name}  (N={N})")
    print(f"{'='*60}")

    results = dict(name=name, N=N, x=x, f=f, M=M, dx=dx)

    # 1. Bernstein operator puro
    m = mse(W_init, M, f)
    results['bernstein_op'] = dict(W=W_init.copy(), mse=m, time_ms=t_bern,
                                   Delta=None)
    print(f"  Bernstein operator      MSE={m:.7f}   t={t_bern:8.2f} ms")

    # 2. SO upper — W_ref = W_bernstein_op
    #    Vincolo: cumsum(Delta)[h] <= 0  per h=0..N-1
    #    → sposta massa verso destra → CDF_BP >= CDF_ref
    t0 = time.perf_counter()
    W_up, D_up = solve_scipy(N, x, f,
                             W_init=W_init.copy(),
                             direction='upper',
                             W_ref=W_init.copy())
    t = (time.perf_counter() - t0) * 1e3 + t_bern
    so_up = check_order(W_up, W_init, 'upper')
    m     = mse(W_up, M, f)
    results['bernstein_op_upper'] = dict(W=W_up, mse=m, time_ms=t,
                                         Delta=D_up, so=so_up,
                                         W_ref=W_init.copy())
    sat = '✓' if so_up['satisfied'] else '✗'
    print(f"  BernOp + SO upper       MSE={m:.7f}   t={t:8.2f} ms  SO:{sat}")

    # 3. SO lower — W_ref = W_bernstein_op
    #    Vincolo: cumsum(Delta)[h] >= 0  per h=0..N-1
    #    → sposta massa verso sinistra → CDF_BP <= CDF_ref
    t0 = time.perf_counter()
    W_lo, D_lo = solve_scipy(N, x, f,
                             W_init=W_init.copy(),
                             direction='lower',
                             W_ref=W_init.copy())
    t = (time.perf_counter() - t0) * 1e3 + t_bern
    so_lo = check_order(W_lo, W_init, 'lower')
    m     = mse(W_lo, M, f)
    results['bernstein_op_lower'] = dict(W=W_lo, mse=m, time_ms=t,
                                          Delta=D_lo, so=so_lo,
                                          W_ref=W_init.copy())
    sat = '✓' if so_lo['satisfied'] else '✗'
    print(f"  BernOp + SO lower       MSE={m:.7f}   t={t:8.2f} ms  SO:{sat}")

    # Riepilogo
    print(f"\n  {'Metodo':<28}  {'MSE':>12}  {'Tempo (ms)':>12}  SO")
    print(f"  {'-'*58}")
    for k in ['bernstein_op', 'bernstein_op_upper', 'bernstein_op_lower']:
        r   = results[k]
        so  = r.get('so', {})
        sat = f"  {'✓' if so.get('satisfied', False) else '✗'}" if so else ''
        print(f"  {k:<28}  {r['mse']:12.7f}  {r['time_ms']:12.1f}{sat}")

    return results


# Plotting

def _cdf_target(f: np.ndarray, dx: float) -> np.ndarray:
    cdf = np.cumsum(np.maximum(f, 0)) * dx
    return cdf / max(cdf[-1], 1e-12)


def plot_bern_pdf(results: dict, save_path: str | None = None):
    """PDF dei tre metodi vs f(x) + barplot MSE."""
    x, f, M = results['x'], results['f'], results['M']
    N, name  = results['N'], results['name']

    fig, (ax_pdf, ax_mse) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Bernstein - PDF  |  {name}  (N={N})',
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
    ax_mse.set_xticklabels(lbls, rotation=25, ha='right', fontsize=8)
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


def plot_bern_cdf(results: dict, save_path: str | None = None):
    """CDF numerica dei tre metodi vs CDF target."""
    x, f, M, dx = results['x'], results['f'], results['M'], results['dx']
    N, name      = results['N'], results['name']
    cdf_t        = _cdf_target(f, dx)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f'Bernstein - CDF  |  {name}  (N={N})',
                 fontsize=12, fontweight='bold')

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


def plot_bern_weights(results: dict, save_path: str | None = None):
    """Istogramma pesi W per i tre metodi."""
    N     = results['N']
    nodes = np.arange(N + 1)
    keys  = list(STYLE.keys())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    fig.suptitle(f'Bernstein - Pesi W  |  {results["name"]}  (N={N})',
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


def plot_bern_delta(results: dict, save_path: str | None = None):
    """
    Per upper e lower: istogramma di Delta[k] (riga superiore)
    e cumsum direzionale (riga inferiore) per verificare il vincolo SO.
    """
    N      = results['N']
    nodes  = np.arange(N + 1)
    so_keys = ['bernstein_op_upper', 'bernstein_op_lower']

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f'Bernstein - Delta = W_new - W_ref  |  '
                 f'{results["name"]}  (N={N})',
                 fontsize=11, fontweight='bold')

    for j, key in enumerate(so_keys):
        st    = STYLE[key]
        Delta = results[key]['Delta']
        so    = results[key].get('so', {})
        sat   = '✓' if so.get('satisfied', False) else '✗'
        direction = 'upper' if 'upper' in key else 'lower'

        # Riga 0: istogramma Delta
        ax_d = axes[0, j]
        colors = [st['color'] if d >= 0 else '#c0392b' for d in Delta]
        ax_d.bar(nodes, Delta, color=colors, alpha=0.75)
        ax_d.axhline(0, color='black', lw=0.8)
        ax_d.set_title(f'{st["label"]}\n'
                       f'|Δ|_max={np.abs(Delta).max():.3f}  '
                       f'sum={Delta.sum():.1e}  SO:{sat}',
                       fontsize=8)
        ax_d.set_xlabel('k')
        ax_d.set_ylabel('Delta[k]')
        ax_d.grid(True, axis='y', alpha=0.3)

        # Riga 1: cumsum direzionale
        ax_c = axes[1, j]
        if direction == 'upper':
            cs    = np.cumsum(Delta)
            label = 'cumsum l→r  [≤ 0 richiesto per k=0..N-1]'
        else:
            cs    = np.cumsum(Delta[::-1])[::-1]   # tail sum
            label = 'cumsum r→l  [≤ 0 richiesto per k=1..N]'

        bar_colors = ['#27ae60' if v <= 1e-4 else '#e74c3c' for v in cs]
        ax_c.bar(nodes, cs, color=bar_colors, alpha=0.65)
        ax_c.axhline(0, color='black', lw=1.0)
        ax_c.set_title(label, fontsize=8)
        ax_c.set_xlabel('k')
        ax_c.set_ylabel('tail-sum Delta[k]')
        ax_c.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_bern_so(results: dict, save_path: str | None = None):
    """
    CDF target + corridoio upper/lower + tutte e tre le CDF Bernstein.
    Mostra chiaramente come upper e lower stringono attorno all'operatore base.
    """
    x, f, M, dx = results['x'], results['f'], results['M'], results['dx']
    N, name      = results['N'], results['name']
    cdf_t        = _cdf_target(f, dx)

    cdf_up = cdf_from_weights(results['bernstein_op_upper']['W'], M, dx)
    cdf_lo = cdf_from_weights(results['bernstein_op_lower']['W'], M, dx)

    so_up_sat = results['bernstein_op_upper']['so'].get('satisfied', False)
    so_lo_sat = results['bernstein_op_lower']['so'].get('satisfied', False)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f'Bernstein - Riepilogo SO  |  {name}  (N={N})\n'
        f'upper: {"✓" if so_up_sat else "✗"}   '
        f'lower: {"✓" if so_lo_sat else "✗"}',
        fontsize=11, fontweight='bold'
    )

    # Corridoio upper/lower
    ax.fill_between(x, cdf_lo, cdf_up,
                    color='#f39c12', alpha=0.12,
                    label='corridoio SO (upper–lower)', zorder=1)

    ax.plot(x, cdf_t, 'k--', lw=2.5, zorder=6, label='CDF target')

    for key, st in STYLE.items():
        cdf_a = cdf_from_weights(results[key]['W'], M, dx)
        ax.plot(x, cdf_a, color=st['color'], ls=st['ls'],
                lw=st['lw'], label=st['label'], alpha=0.88, zorder=3)

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


# ---------------------------------------------------
#  Esperimento 1 — Beta(4,10): unimodale, moda ~0.27
# ---------------------------------------------------
f1 = stats.beta(4, 10).pdf(x)

r1 = run_bernstein('Beta(4,10) - unimodale', N, x, f1)
plot_bern_pdf(r1, save_path=f'{OUT}/exp1_bern_pdf.png')
plot_bern_cdf(r1, save_path=f'{OUT}/exp1_bern_cdf.png')
plot_bern_weights(r1, save_path=f'{OUT}/exp1_bern_weights.png')
plot_bern_delta(r1, save_path=f'{OUT}/exp1_bern_delta.png')
plot_bern_so(r1, save_path=f'{OUT}/exp1_bern_so.png')
print("  → Esperimento 1 completato.")

# ----------------------------------------------------
#  Esperimento 2 — Polinomio x^3(1-x)^2, moda in x=0.6
# ----------------------------------------------------
f2_raw = x ** 3 * (1 - x) ** 2
f2     = f2_raw / np.trapezoid(f2_raw, x)

r2 = run_bernstein('Polinomio x³(1-x)²  (moda=0.6)', N, x, f2)
plot_bern_pdf(r2, save_path=f'{OUT}/exp2_bern_pdf.png')
plot_bern_cdf(r2, save_path=f'{OUT}/exp2_bern_cdf.png')
plot_bern_weights(r2, save_path=f'{OUT}/exp2_bern_weights.png')
plot_bern_delta(r2, save_path=f'{OUT}/exp2_bern_delta.png')
plot_bern_so(r2, save_path=f'{OUT}/exp2_bern_so.png')
print("  → Esperimento 2 completato.")

# ----------------------------------------------------
#  Esperimento 3 — Beta mixture bimodale
#  0.5*Beta(3,10) + 0.5*Beta(10,3), moda ~0.18 e ~0.82
# -----------------------------------------------------
f3 = 0.5 * stats.beta(3, 10).pdf(x) + 0.5 * stats.beta(10, 3).pdf(x)

f3 = f3 / np.trapezoid(f3, x)

r3 = run_bernstein('Beta mixture bimodale', N, x, f3)
plot_bern_pdf(r3, save_path=f'{OUT}/exp3_bern_pdf.png')
plot_bern_cdf(r3, save_path=f'{OUT}/exp3_bern_cdf.png')
plot_bern_weights(r3, save_path=f'{OUT}/exp3_bern_weights.png')
plot_bern_delta(r3, save_path=f'{OUT}/exp3_bern_delta.png')
plot_bern_so(r3, save_path=f'{OUT}/exp3_bern_so.png')
print("  → Esperimento 3 (bimodale) completato.")

"""
bernstein/plotting.py

Funzioni di visualizzazione:

  plot_pdf_comparison  : PDF di tutti i metodi + barplot MSE
  plot_cdf_all         : CDF di tutti i metodi vs CDF target
  plot_weights         : istogramma pesi W
  plot_delta           : istogramma Delta per i metodi SO
  plot_so_single       : CDF di un singolo metodo SO con evidenziazione
  plot_so_summary      : tutte le CDF SO su un pannello con corridoio
"""

from bernstein.basis import cdf_from_weights
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')


# Palette

STYLE = {
    'bernstein_op':       dict(color='#95a5a6', ls='--', lw=1.6, label='Bernstein operator'),
    'bernstein_op_upper': dict(color='#f39c12', ls=':',  lw=1.8, label='BernOp + SO upper'),
    'bernstein_op_lower': dict(color='#1abc9c', ls=':',  lw=1.8, label='BernOp + SO lower'),
    'ing_cane':           dict(color='#e74c3c', ls='-',  lw=1.8, label='Ing-Cane'),
    'scipy':              dict(color='#27ae60', ls='-',  lw=2.0, label='Scipy SLSQP'),
    'scipy_upper':        dict(color='#e67e22', ls='-',  lw=1.8, label='Scipy + SO upper'),
    'scipy_lower':        dict(color='#8e44ad', ls='-',  lw=1.8, label='Scipy + SO lower'),
    'pytorch':            dict(color='#2980b9', ls='--', lw=1.8, label='PyTorch'),
    'pytorch_upper':      dict(color='#c0392b', ls='--', lw=1.6, label='PyTorch + SO (pen.)'),
}


def _cdf_target(f: np.ndarray, dx: float) -> np.ndarray:
    cdf = np.cumsum(np.maximum(f, 0)) * dx
    return cdf / max(cdf[-1], 1e-12)


# 1. Confronto PDF

def plot_pdf_comparison(results: dict, save_path: str | None = None):
    """PDF di tutti i metodi vs f(x) originale + barplot MSE con tempi."""
    x = results['x']
    f = results['f']
    M = results['M']
    N = results['N']
    name = results['name']

    fig, (ax_pdf, ax_mse) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Approssimazione PDF - {name}  (N={N})',
                 fontsize=13, fontweight='bold')

    ax_pdf.plot(x, f, 'k--', lw=2.5, zorder=6, label='f(x) target')
    for key, st in STYLE.items():
        if key not in results:
            continue
        ax_pdf.plot(x, M @ results[key]['W'], **st, alpha=0.85)

    ax_pdf.set_xlabel('x')
    ax_pdf.set_ylabel('Densità')
    ax_pdf.set_title('PDF')
    ax_pdf.legend(fontsize=7.5)
    ax_pdf.grid(True, alpha=0.3)

    keys = [k for k in STYLE if k in results]
    mses = [results[k]['mse'] for k in keys]
    times = [results[k]['time_ms'] for k in keys]
    cols = [STYLE[k]['color'] for k in keys]
    lbls = [STYLE[k]['label'] for k in keys]

    bars = ax_mse.bar(range(len(keys)), mses, color=cols, alpha=0.8)
    ax_mse.set_xticks(range(len(keys)))
    ax_mse.set_xticklabels(lbls, rotation=40, ha='right', fontsize=7)
    ax_mse.set_ylabel('MSE')
    ax_mse.set_title('MSE per metodo')
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


# 2. CDF di tutti i metodi

def plot_cdf_all(results: dict, save_path: str | None = None):
    """CDF calcolata numericamente per ogni metodo sovrapposta alla CDF target."""
    x = results['x']
    f = results['f']
    M = results['M']
    dx = results['dx']
    N = results['N']
    name = results['name']

    cdf_t = _cdf_target(f, dx)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f'CDF - tutti i metodi  |  {name}  (N={N})',
                 fontsize=12, fontweight='bold')

    ax.plot(x, cdf_t, 'k--', lw=2.5, zorder=6, label='CDF target')

    for key, st in STYLE.items():
        if key not in results:
            continue
        cdf_a = cdf_from_weights(results[key]['W'], M, dx)
        ax.plot(x, cdf_a, color=st['color'], ls=st['ls'],
                lw=st['lw'], label=st['label'], alpha=0.85)

    ax.set_xlabel('x')
    ax.set_ylabel('CDF')
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# 3. Istogramma pesi

def plot_weights(results: dict, save_path: str | None = None):
    """Istogramma pesi W per i principali metodi (escluso ing_cane per scala)."""
    N = results['N']
    nodes = np.arange(N + 1)
    keys = [k for k in ['bernstein_op', 'scipy', 'scipy_upper', 'scipy_lower',
                        'bernstein_op_upper', 'bernstein_op_lower', 'pytorch']
            if k in results]

    fig, axes = plt.subplots(1, len(keys), figsize=(len(keys) * 3.2, 4),
                             sharey=True)
    if len(keys) == 1:
        axes = [axes]

    fig.suptitle(f'Pesi W — {results["name"]}  (N={N})',
                 fontsize=11, fontweight='bold')

    for ax, key in zip(axes, keys):
        st = STYLE[key]
        W = results[key]['W']
        ax.bar(nodes, W, color=st['color'], alpha=0.75)
        ax.set_title(f'{st["label"]}\nMSE={
                     results[key]["mse"]:.5f}', fontsize=7.5)
        ax.set_xlabel('k')
        ax.grid(True, axis='y', alpha=0.3)

    axes[0].set_ylabel('W[k]')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# 4. Istogramma Delta

def plot_delta(results: dict, save_path: str | None = None):
    """
    Istogramma di Delta = W_new - W_ref per ogni metodo con vincolo SO.
    Mostra anche cumsum(Delta) per verificare visivamente il vincolo.
    """
    N = results['N']
    nodes = np.arange(N + 1)
    so_keys = [k for k in ['bernstein_op_upper', 'bernstein_op_lower',
                            'scipy_lower', 'scipy_upper', 'pytorch_upper']
               if k in results and results[k].get('Delta') is not None]

    if not so_keys:
        return

    fig, axes = plt.subplots(2, len(so_keys),
                             figsize=(len(so_keys) * 3.5, 7))
    if len(so_keys) == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f'Delta = W_new - W_ref  |  {results["name"]}  (N={N})',
                 fontsize=11, fontweight='bold')

    for j, key in enumerate(so_keys):
        st = STYLE[key]
        Delta = results[key]['Delta']
        so = results[key].get('so', {})
        sat = '✓' if so.get('satisfied', False) else '✗'

        # Riga superiore: istogramma Delta
        ax_d = axes[0, j]
        colors = [st['color'] if d > 0 else '#c0392b' for d in Delta]
        ax_d.bar(nodes, Delta, color=colors, alpha=0.75)
        ax_d.axhline(0, color='black', lw=0.8)
        ax_d.set_title(f'{st["label"]}\n'
                       f'|Δ|_max={np.abs(Delta).max():.3f}  sum={
            Delta.sum():.1e}  SO:{sat}',
            fontsize=7.5)
        ax_d.set_xlabel('k')
        ax_d.set_ylabel('Delta[k]')
        ax_d.grid(True, axis='y', alpha=0.3)

        # Riga inferiore: cumsum direzionale di Delta
        ax_c = axes[1, j]
        direction = 'upper' if 'upper' in key else 'lower'
        if direction == 'upper':
            # l->r: sum(Delta[0:k+1]) deve essere ≤ 0
            cs = np.cumsum(Delta)
            thresh_label = 'cumsum l→r   [≤ 0 richiesto per k=0..N-1]'
        else:
            # r->l: sum(Delta[k:]) deve essere ≤ 0
            cs = np.cumsum(Delta[::-1])[::-1]   # cs[k] = sum(Delta[k:])
            thresh_label = 'cumsum r→l   [≤ 0 richiesto per k=1..N]'
        ax_c.bar(nodes, cs, color=st['color'], alpha=0.55)
        ax_c.axhline(0, color='black', lw=1.0)
        ax_c.set_title(thresh_label, fontsize=7.5)
        ax_c.set_xlabel('k')
        ax_c.set_ylabel('tail-sum Delta[k]')
        ax_c.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# 5. SO singolo metodo

def plot_so_single(results: dict, key: str, direction: str,
                   save_path: str | None = None):
    """
    Due pannelli affiancati per un singolo metodo con vincolo SO:
      (sx) CDF completa: target vs approssimante
      (dx) Stessa CDF zoomata nella zona di margine minimo
    """
    x = results['x']
    f = results['f']
    M = results['M']
    dx = results['dx']
    N = results['N']
    W = results[key]['W']
    st = STYLE[key]
    so = results[key].get('so', {})
    name = results['name']

    cdf_t = _cdf_target(f, dx)
    cdf_a = cdf_from_weights(W, M, dx)
    delta_cdf = cdf_a - cdf_t

    idx_crit = np.argmin(np.abs(delta_cdf))
    x_crit = x[idx_crit]
    zoom_hw = 0.15
    x_lo = max(x[0], x_crit - zoom_hw)
    x_hi = min(x[-1], x_crit + zoom_hw)
    mask = (x >= x_lo) & (x <= x_hi)

    sat_str = '✓ Soddisfatto' if so.get('satisfied', False) else '✗ Violato'
    dir_str = 'upper  (F_approx ≥ F_target)' if direction == 'upper' \
              else 'lower  (F_approx ≤ F_target)'

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f'SO {dir_str} — {st["label"]}\n'
        f'{name}  (N={N})   {sat_str}   '
        f'margine min = {np.abs(delta_cdf).min():.4f}',
        fontsize=10, fontweight='bold'
    )

    for ax, xdata, cdf_t_data, cdf_a_data, title in [
        (ax_full, x,       cdf_t,       cdf_a,       'Vista completa'),
        (ax_zoom, x[mask], cdf_t[mask], cdf_a[mask],
         f'Zoom x∈[{x_lo:.2f},{x_hi:.2f}]'),
    ]:
        ax.plot(xdata, cdf_t_data, 'k--', lw=2.2, label='CDF target')
        ax.plot(xdata, cdf_a_data, color=st['color'], lw=2,
                label=f'CDF {st["label"]}')

        if direction == 'upper':
            ax.fill_between(xdata, cdf_t_data, cdf_a_data,
                            where=(cdf_a_data >= cdf_t_data),
                            color=st['color'], alpha=0.18,
                            label='F_approx ≥ F_target ✓')
        else:
            ax.fill_between(xdata, cdf_a_data, cdf_t_data,
                            where=(cdf_a_data <= cdf_t_data),
                            color=st['color'], alpha=0.18,
                            label='F_approx ≤ F_target ✓')

        ax.set_xlabel('x')
        ax.set_ylabel('CDF')
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_full.axvspan(x_lo, x_hi, color='gray', alpha=0.10, zorder=0,
                    label='zona zoom')
    ax_full.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# 6. SO riepilogo tutti i metodi

def plot_so_summary(results: dict, save_path: str | None = None):
    """
    Tutte le CDF SO sovrapposte alla CDF target, con corridoio upper/lower.
    Mostra sia i metodi scipy (vincolo esatto) che bernstein_op (vincolo esatto)
    e pytorch (vincolo soft/penalità).
    """
    x = results['x']
    f = results['f']
    M = results['M']
    dx = results['dx']
    N = results['N']
    name = results['name']

    cdf_t = _cdf_target(f, dx)

    so_keys = [k for k in ['bernstein_op', 'bernstein_op_upper', 'bernstein_op_lower',
                           'scipy', 'scipy_upper', 'scipy_lower',
                           'pytorch', 'pytorch_upper']
               if k in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f'Riepilogo Ordine Stocastico - {name}  (N={N})',
                 fontsize=12, fontweight='bold')

    ax.plot(x, cdf_t, 'k--', lw=2.5, zorder=6, label='CDF target')

    # Corridoio scipy upper/lower
    if 'scipy_upper' in results and 'scipy_lower' in results:
        cdf_up = cdf_from_weights(results['scipy_upper']['W'], M, dx)
        cdf_lo = cdf_from_weights(results['scipy_lower']['W'], M, dx)
        ax.fill_between(x, cdf_lo, cdf_up, color='#27ae60', alpha=0.08,
                        label='corridoio SO scipy (upper–lower)', zorder=1)

    # Corridoio bernstein_op upper/lower (se presente)
    if 'bernstein_op_upper' in results and 'bernstein_op_lower' in results:
        cdf_bo_up = cdf_from_weights(results['bernstein_op_upper']['W'], M, dx)
        cdf_bo_lo = cdf_from_weights(results['bernstein_op_lower']['W'], M, dx)
        ax.fill_between(x, cdf_bo_lo, cdf_bo_up, color='#f39c12', alpha=0.07,
                        label='corridoio SO BernOp (upper–lower)', zorder=1)

    for key in so_keys:
        st = STYLE[key]
        cdf_a = cdf_from_weights(results[key]['W'], M, dx)
        ax.plot(x, cdf_a, color=st['color'], ls=st['ls'],
                lw=st['lw'], label=st['label'], alpha=0.88, zorder=3)

    ax.set_xlabel('x')
    ax.set_ylabel('CDF')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

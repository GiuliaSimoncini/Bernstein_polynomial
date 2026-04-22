"""
bernstein/experiments.py

Runner completo. Per ogni metodo registra W, Delta (se SO), MSE, tempo.

Metodi eseguiti:

  bernstein_op        : operatore di Bernstein classico (no ottimizzazione)
  bernstein_op_upper  : SO upper partendo da W_bernstein_op (W_ref = W_bernstein_op)
  bernstein_op_lower  : SO lower partendo da W_bernstein_op (W_ref = W_bernstein_op)
  ing_cane            : soluzione quasi-chiusa O(n)
  scipy               : SLSQP non vincolato (ottimo globale)
  scipy_upper         : SO upper rispetto a W_scipy (W_ref = W_scipy)
  scipy_lower         : SO lower rispetto a W_scipy (W_ref = W_scipy)
  pytorch             : gradient descent non vincolato
  pytorch_upper       : gradient descent con penalità SO upper (W_ref = W_scipy)

Logica dei vincoli SO

Il vincolo SO è sempre definito in termini di Delta = W_new - W_ref:

  upper:  cumsum(Delta)[h] <= 0  ∀h=0..N-1   → massa verso destra   → BP >=_st ref
  lower:  cumsum(Delta)[h] >= 0  ∀h=0..N-1   → massa verso sinistra → BP <=_st ref

Per bernstein_op_{upper,lower}: W_ref = W_bernstein_op  (SO rispetto all'operatore)
Per scipy_{upper,lower}:        W_ref = W_scipy         (SO rispetto all'ottimo)
Per pytorch_upper:              W_ref = W_scipy         (idem, con penalità soft)

sum(Delta) = 0 è garantito dal vincolo di misura unitaria:
    sum(W_new) = N+1  e  sum(W_ref) = N+1  →  sum(Delta) = 0
"""

import numpy as np
import time

from bernstein.basis import basis_matrix, bernstein_operator_init, mse
from bernstein.stochastic import check_order, compute_delta
from bernstein.methods import (solve_ing_cane, solve_scipy,
                               solve_pytorch, solve_pytorch_ordered)


# serve per il print scipy
def _fmt_vec(v: np.ndarray, decimals: int = 3) -> str:
    """Formatta un vettore numpy su una riga."""
    return "[" + "  ".join(f"{x:+.{decimals}f}" for x in v) + "]"


# console spam utile per capire i delta e W
def _print_w_delta(label: str, W: np.ndarray, Delta: np.ndarray | None = None):
    """Stampa W e Delta in modo compatto."""
    print(f"\n    -- {label}")
    print(f"    W    = {_fmt_vec(W)}")
    print(f"    sum(W) = {W.sum():.6f}")
    if Delta is not None:
        print(f"    Delta = {_fmt_vec(Delta)}")
        print(f"    sum(Delta) = {Delta.sum():.2e}   "
              f"|Delta|_max = {np.abs(Delta).max():.4f}   "
              f"|Delta|_sum_abs = {np.abs(Delta).sum():.4f}")
        cs = np.cumsum(Delta)[:-1]
        cs_ok_upper = (cs <= 1e-4).all()
        cs_ok_lower = (cs >= -1e-4).all()
        print(f"    cumsum(Delta)[:-1] min={cs.min():+.4f}  max={cs.max():+.4f}  "
              f"(<=0: {'✓' if cs_ok_upper else '✗'}  >=0: {'✓' if cs_ok_lower else '✗'})")


def run_experiment(name: str, N: int, x: np.ndarray, f: np.ndarray,
                   epochs_pt: int = 3000) -> dict:

    # Crea base di Bernstein
    M = basis_matrix(N, x)
    dx = x[1] - x[0]

    # Inizializza i pesi
    t0 = time.perf_counter()
    W_init  = bernstein_operator_init(N, x, f)
    t_bern = (time.perf_counter() - t0) * 1e3

    print(f"\n{'='*70}")
    print(f"  {name}  (N={N})")
    print(f"{'='*70}")

    results = dict(name=name, N=N, x=x, f=f, M=M, dx=dx)

    def _store(label, W, elapsed, Delta=None):
        m = mse(W, M, f)
        results[label] = dict(W=W, mse=m, time_ms=elapsed, Delta=Delta)
        return m

    # Bernstein operator (nessuna ottimizzazione)
    m = _store('bernstein_op', W_init.copy(), t_bern, Delta=None)
    print(f"\n  Bernstein operator      MSE={m:.7f}   t={t_bern:8.2f} ms")
    _print_w_delta('bernstein_op', W_init)

    
    # Bernstein operator + SO upper
    # Variabile ottimizzata: Delta = W_new - W_bernstein_op
    # Vincoli: cumsum(Delta)[h] <= 0  per ogni h
    t0 = time.perf_counter()
    W_bo_up, D_bo_up = solve_scipy(N, x, f,
                                   W_init=W_init.copy(),
                                   direction='upper',
                                   W_ref=W_init.copy())
    t = (time.perf_counter() - t0) * 1e3 + t_bern
    so = check_order(W_bo_up, W_init, 'upper')
    m = _store('bernstein_op_upper', W_bo_up, t, Delta=D_bo_up)
    results['bernstein_op_upper']['so'] = so
    results['bernstein_op_upper']['W_ref'] = W_init.copy()
    sat = '✓' if so['satisfied'] else '✗'
    print(f"  BernOp + SO upper       MSE={m:.7f}   t={t:8.2f} ms  SO:{sat}")
    _print_w_delta('bernstein_op_upper', W_bo_up, D_bo_up)

    # Bernstein operator + SO lower
    # Stessa cosa di prima semplicemente si fa dalla direzione opposta
    t0 = time.perf_counter()
    W_bo_lo, D_bo_lo = solve_scipy(N, x, f,
                                   W_init=W_init.copy(),
                                   direction='lower',
                                   W_ref=W_init.copy())
    t = (time.perf_counter() - t0) * 1e3 + t_bern
    so = check_order(W_bo_lo, W_init, 'lower')
    m = _store('bernstein_op_lower', W_bo_lo, t, Delta=D_bo_lo)
    results['bernstein_op_lower']['so'] = so
    results['bernstein_op_lower']['W_ref'] = W_init.copy()
    sat = '✓' if so['satisfied'] else '✗'
    print(f"  BernOp + SO lower       MSE={m:.7f}   t={t:8.2f} ms  SO:{sat}")
    _print_w_delta('bernstein_op_lower', W_bo_lo, D_bo_lo)
    
    # Ing-Cane
    # Niente SO perché ing-cane non può dare garanzie su questo
    t0 = time.perf_counter()
    W_ic = solve_ing_cane(N, x, f)
    t = (time.perf_counter() - t0) * 1e3
    m = _store('ing_cane', W_ic, t)
    print(f"\n  Ing-Cane                MSE={m:.7f}   t={t:8.2f} ms")
    _print_w_delta('ing_cane', W_ic)
    
    # Scipy unconstrained
    # Scipy senza SO dovrebbe sempre essere il migliore (MSE = 0)
    t0 = time.perf_counter()
    W_scipy, _ = solve_scipy(N, x, f, W_init=W_init.copy())
    t_scipy = (time.perf_counter() - t0) * 1e3
    m = _store('scipy', W_scipy, t_scipy, Delta=None)
    print(f"\n  Scipy SLSQP             MSE={m:.7f}   t={t_scipy:8.2f} ms")
    _print_w_delta('scipy', W_scipy)

    
    # Scipy + SO upper
    # Variabile: Delta = W_new - W_scipy.
    # Soglia SO: cumsum(W_new)[h] <= (N+1)*F_f(h/N) - rispetto alla CDF di f.
    # Se W_scipy ≈ f, la soglia coincide con cumsum(W_scipy) e si forza
    # uno shift reale verso destra.
    t0 = time.perf_counter()
    W_up, D_up = solve_scipy(N, x, f,
                             W_init=W_scipy.copy(),
                             direction='upper',
                             W_ref=W_scipy.copy())
    t = (time.perf_counter() - t0) * 1e3 + t_scipy
    so_up = check_order(W_up, W_scipy, 'upper')
    m = _store('scipy_upper', W_up, t, Delta=D_up)
    results['scipy_upper']['so'] = so_up
    results['scipy_upper']['W_ref'] = W_scipy.copy()
    sat = '✓' if so_up['satisfied'] else '✗'
    print(f"\n  Scipy + SO upper        MSE={m:.7f}   t={t:8.2f} ms  SO:{sat}")
    _print_w_delta('scipy_upper', W_up, D_up)

    # Scipy + SO lower - come sopra, direzione opposta
    t0 = time.perf_counter()
    W_lo, D_lo = solve_scipy(N, x, f,
                             W_init=W_scipy.copy(),
                             direction='lower',
                             W_ref=W_scipy.copy())
    t = (time.perf_counter() - t0) * 1e3 + t_scipy
    so_lo = check_order(W_lo, W_scipy, 'lower')
    m = _store('scipy_lower', W_lo, t, Delta=D_lo)
    results['scipy_lower']['so'] = so_lo
    results['scipy_lower']['W_ref'] = W_scipy.copy()
    sat = '✓' if so_lo['satisfied'] else '✗'
    print(f"\n  Scipy + SO lower        MSE={m:.7f}   t={t:8.2f} ms  SO:{sat}")
    _print_w_delta('scipy_lower', W_lo, D_lo)
    
    
    # PyTorch unconstrained
    # fa gradient descent
    t0 = time.perf_counter()
    W_pt = solve_pytorch(N, x, f, epochs=epochs_pt)
    t_pytorch = (time.perf_counter() - t0) * 1e3
    m = _store('pytorch', W_pt, t_pytorch)
    print(f"\n  PyTorch                 MSE={m:.7f}   t={t_pytorch:8.2f} ms")
    _print_w_delta('pytorch', W_pt)

    # PyTorch + SO upper (W_ref = W_scipy)
    # Aggiunge ordinamento stocastico, per farlo usa delle penalità
    # quando l'approssimazione non rispetta l'ordinamento.
    # In questo modo il gradient descent riesce ad auto bilanciarsi
    # e trovare i pesi per far funzionare tutto
    t0 = time.perf_counter()
    W_pt_up, D_pt_up = solve_pytorch_ordered(N, x, f, W_scipy.copy(),
                                             direction='upper')
    t = (time.perf_counter() - t0) * 1e3 + t_pytorch
    so = check_order(W_pt_up, W_scipy, 'upper')
    m = _store('pytorch_upper', W_pt_up, t, Delta=D_pt_up)
    results['pytorch_upper']['so'] = so
    results['pytorch_upper']['W_ref'] = W_scipy.copy()
    sat = '✓' if so['satisfied'] else '✗'
    print(f"\n  PyTorch + SO (pen.)     MSE={m:.7f}   t={t:8.2f} ms  SO:{sat}")
    _print_w_delta('pytorch_upper', W_pt_up, D_pt_up)

    # Riepilogo
    print(f"\n{'─'*70}")
    method_order = ['bernstein_op', 'bernstein_op_upper', 'bernstein_op_lower',
                    'ing_cane', 'scipy', 'scipy_upper', 'scipy_lower',
                    'pytorch', 'pytorch_upper']
    print(f"  {'Metodo':<28}  {'MSE':>12}  {'Tempo (ms)':>12}  SO")
    print(f"  {'-'*60}")
    for k in method_order:
        if k not in results:
            continue
        m = results[k]['mse']
        t = results[k]['time_ms']
        so_info = results[k].get('so', {})
        sat = f"  {'✓' if so_info.get(
            'satisfied', False) else '✗'}" if so_info else ''
        print(f"  {k:<28}  {m:12.7f}  {t:12.1f}{sat}")

    return results

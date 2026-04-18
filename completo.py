"""
main.py

Esegue due esperimenti e produce tutti i grafici.

Esperimento 1: Beta(4,10)         — PDF unimodale asimmetrica (moda ~0.27)
Esperimento 2: x^3 (1-x)^2       — polinomio grado 5, moda in x=0.6

Per ciascun esperimento vengono prodotti:
  expN_pdf.png        : confronto PDF + barplot MSE
  expN_cdf.png        : CDF di tutti i metodi vs CDF target
  expN_weights.png    : istogramma pesi W
  expN_delta.png      : istogramma Delta per i metodi SO
  expN_bo_so_upper.png: SO upper BernOp (CDF + zoom)
  expN_bo_so_lower.png: SO lower BernOp
  expN_so_summary.png : tutte le CDF SO + corridoi
"""

import os
import numpy as np
import scipy.stats as stats

from bernstein.experiments import run_experiment
from bernstein.plotting import (plot_pdf_comparison, plot_cdf_all,
                                plot_weights, plot_delta,
                                plot_so_single, plot_so_summary)

OUT = 'immagini_completo'
os.makedirs(OUT, exist_ok=True)

N = 30
x = np.linspace(0.001, 0.999, 400)

# ----------------------------------------------------
#  Esperimento 1 - Beta(4,10): unimodale, moda ~0.27
# ----------------------------------------------------
f1 = stats.beta(4, 10).pdf(x)
r1 = run_experiment('Beta(4,10) - unimodale', N, x, f1, epochs_pt=3000)

plot_pdf_comparison(r1, save_path=f'{OUT}/exp1_pdf.png')
plot_cdf_all(r1, save_path=f'{OUT}/exp1_cdf.png')
plot_weights(r1, save_path=f'{OUT}/exp1_weights.png')
plot_delta(r1, save_path=f'{OUT}/exp1_delta.png')
 
plot_so_single(r1, 'scipy_upper','upper',
               save_path=f'{OUT}/exp1_so_upper.png')
plot_so_single(r1, 'scipy_lower', 'lower',
               save_path=f'{OUT}/exp1_so_lower.png')
plot_so_single(r1, 'bernstein_op_upper', 'upper',
               save_path=f'{OUT}/exp1_bo_so_upper.png')
plot_so_single(r1, 'bernstein_op_lower', 'lower',
               save_path=f'{OUT}/exp1_bo_so_lower.png')
plot_so_summary(r1, save_path=f'{OUT}/exp1_so_summary.png')
print("  → Esperimento 1 completato.")

# --------------------------------------------------------------
#  Esperimento 2 — Polinomio x^3(1-x)^2, grado 5, moda in x=0.6
# --------------------------------------------------------------
f2_raw = x ** 3 * (1 - x) ** 2
f2 = f2_raw / np.trapezoid(f2_raw, x)

r2 = run_experiment('Polinomio x³(1-x)²  (moda=0.6)', N, x, f2, epochs_pt=5000)

plot_pdf_comparison(r2, save_path=f'{OUT}/exp2_pdf.png')
plot_cdf_all(r2, save_path=f'{OUT}/exp2_cdf.png')
plot_weights(r2, save_path=f'{OUT}/exp2_weights.png')
plot_delta(r2, save_path=f'{OUT}/exp2_delta.png')
plot_so_single(r2, 'scipy_upper', 'upper',
               save_path=f'{OUT}/exp2_so_upper.png')
plot_so_single(r2, 'scipy_lower', 'lower',
               save_path=f'{OUT}/exp2_so_lower.png')
plot_so_single(r2, 'bernstein_op_upper', 'upper',
               save_path=f'{OUT}/exp2_bo_so_upper.png')
plot_so_single(r2, 'bernstein_op_lower', 'lower',
               save_path=f'{OUT}/exp2_bo_so_lower.png')
plot_so_summary(r2, save_path=f'{OUT}/exp2_so_summary.png')
print("  → Esperimento 2 completato.")

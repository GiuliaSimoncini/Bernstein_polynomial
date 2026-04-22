import os
import time
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------
# FUNZIONI DI BASE
# ----------------

def basis_matrix_stable(N: int, x: np.ndarray) -> np.ndarray:
    """
    Matrice della base di Bernstein standard di grado N, stabile per N grandi.
    Sfrutta la PMF della distribuzione binomiale per evitare overflow numerici.
    """
    k = np.arange(N + 1)
    M = stats.binom.pmf(k[np.newaxis, :], N, x[:, np.newaxis])
    return M

def bernstein_operator_init(N: int, x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Operatore di Bernstein classico: campiona la funzione nei nodi k/N.
    Garantisce che la somma dei pesi faccia N+1.
    """
    nodes = np.linspace(0, 1, N + 1)
    W = np.interp(nodes, x, f)
    W = np.maximum(W, 0.0)
    total = W.sum()
    return W * (N + 1) / total if total > 1e-12 else np.ones(N + 1)

def mse(W: np.ndarray, M: np.ndarray, f: np.ndarray) -> float:
    return float(np.mean((M @ W - f)**2))

def cdf_from_weights(W: np.ndarray, M: np.ndarray, dx: float) -> np.ndarray:
    pdf = np.maximum(M @ W, 0.0)
    cdf = np.cumsum(pdf) * dx
    return cdf / max(cdf[-1], 1e-12)

def _cdf_target(f: np.ndarray, dx: float) -> np.ndarray:
    cdf = np.cumsum(np.maximum(f, 0)) * dx
    return cdf / max(cdf[-1], 1e-12)

# --------------------------
# CONFIGURAZIONE ESPERIMENTI
# --------------------------

OUT = 'immagini_bern_base_alternativa_n_grande'
os.makedirs(OUT, exist_ok=True)

# Lista dei gradi N da testare nel ciclo
N_LIST = [50, 1000, 5000, 10000, 50000, 100000, 500000] 
x = np.linspace(0.001, 0.999, 500)
dx = x[1] - x[0]

# -----------------
# RUNNER E PLOTTING
# -----------------

def run_experiment(name: str, exp_id: int, target_f: np.ndarray):
    print(f"\n{'='*60}")
    print(f" ESPERIMENTO {exp_id}: {name}")
    print(f"{'='*60}")
    
    target_cdf = _cdf_target(target_f, dx)
    results = {}
    
    # 1. CICLO SUI VARI GRADI N
    for n in N_LIST:
        t0 = time.perf_counter()
        
        M = basis_matrix_stable(n, x)
        W = bernstein_operator_init(n, x, target_f)
        approx_pdf = M @ W
        approx_cdf = cdf_from_weights(W, M, dx)
        
        calc_time = (time.perf_counter() - t0) * 1000
        error = mse(W, M, target_f)
        
        results[n] = {
            'pdf': approx_pdf,
            'cdf': approx_cdf,
            'mse': error,
            'time': calc_time
        }
        
        # MSE stampato con 10 decimali
        print(f" Grado N={n:<7} | MSE={error:.10f} | Tempo={calc_time:>6.2f} ms")

    # 2. GRAFICO PDF
    plt.figure(figsize=(10, 6))
    plt.plot(x, target_f, 'k--', lw=2.5, label='f(x) Target', zorder=10)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(N_LIST)))
    for (n, res), color in zip(results.items(), colors):
        # 10 decimali per l'MSE nella legenda
        plt.plot(x, res['pdf'], color=color, lw=2, alpha=0.8, 
                 label=f'Bernstein N={n} (MSE: {res["mse"]:.10f})')
        
    plt.title(f'Convergenza PDF - {name}', fontsize=14, fontweight='bold')
    plt.xlabel('x')
    plt.ylabel('Densità')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/exp{exp_id}_{name.replace(" ", "_")}_pdf.png', dpi=150)
    plt.close()

    # 3. GRAFICO CDF
    plt.figure(figsize=(10, 6))
    plt.plot(x, target_cdf, 'k--', lw=2.5, label='CDF Target', zorder=10)
    
    for (n, res), color in zip(results.items(), colors):
        plt.plot(x, res['cdf'], color=color, lw=2, alpha=0.8, label=f'Bernstein N={n}')
        
    plt.title(f'Convergenza CDF - {name}', fontsize=14, fontweight='bold')
    plt.xlabel('x')
    plt.ylabel('Probabilità Cumulata')
    plt.ylim(-0.02, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/exp{exp_id}_{name.replace(" ", "_")}_cdf.png', dpi=150)
    plt.close()
    
    # 4. LINE PLOT: MSE vs GRADO N
    plt.figure(figsize=(8, 4))
    n_vals = list(results.keys())
    mse_vals = [res['mse'] for res in results.values()]
    time_vals = [res['time'] for res in results.values()]
    
    plt.plot(n_vals, mse_vals, 'o-', color='#e74c3c', lw=2, markersize=8)
    plt.title(f'Decadimento dell\'Errore (MSE) - {name}', fontsize=12, fontweight='bold')
    plt.xlabel('Grado N')
    plt.ylabel('MSE')
    plt.yscale('log') 
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{OUT}/exp{exp_id}_{name.replace(" ", "_")}_mse_line.png', dpi=150)
    plt.close()

    # 5. BAR PLOT: ISTOGRAMMA MSE CON TEMPI SOPRA LE BARRE
    plt.figure(figsize=(8, 5))
    n_str_vals = [str(n) for n in n_vals]
    
    bars = plt.bar(n_str_vals, mse_vals, color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.5)
    plt.title(f'MSE per Grado N (con tempo di esecuzione) - {name}', fontsize=12, fontweight='bold')
    plt.xlabel('Grado N')
    plt.ylabel('MSE (Scala Logaritmica)')
    plt.yscale('log') # Indispensabile per vedere la differenza tra le barre
    plt.grid(True, axis='y', which="both", ls="--", alpha=0.3)
    
    # Aggiunge il tempo in ms sopra ogni barra
    for bar, t in zip(bars, time_vals):
        height = bar.get_height()
        # In scala logaritmica, per distanziare il testo verso l'alto moltiplichiamo
        y_pos = height * 1.25 
        label = f'{t:.1f} ms' if t >= 0.1 else '<0.1 ms'
        plt.text(bar.get_x() + bar.get_width() / 2, y_pos,
                 label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}/exp{exp_id}_{name.replace(" ", "_")}_mse_bar.png', dpi=150)
    plt.close()

# ----------
# ESECUZIONE
# ----------

if __name__ == "__main__":
    print("Inizio calcolo matrici e approssimazioni...")
    
    # Esperimento 1
    f1 = stats.beta(4, 10).pdf(x)
    run_experiment('Beta_Unimodale', 1, f1)
    
    # Esperimento 2
    f2_raw = x**3 * (1 - x)**2
    f2 = f2_raw / np.trapezoid(f2_raw, x)
    run_experiment('Polinomio', 2, f2)
    
    # Esperimento 3
    f3 = 0.5 * stats.beta(3, 10).pdf(x) + 0.5 * stats.beta(10, 3).pdf(x)
    run_experiment('Beta_Bimodale', 3, f3)
    
    print(f"\n Tutti i grafici sono stati salvati nella cartella '{OUT}'.")
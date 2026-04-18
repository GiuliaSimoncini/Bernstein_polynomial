"""
bernstein/stochastic.py

Vincoli di ordinamento stocastico del primo ordine - formulazione Delta.

Formulazione del problema:

Dati pesi di riferimento W_ref (tipicamente la soluzione ottima senza vincoli SO,
o i pesi dell'operatore di Bernstein), vogliamo trovare la correzione

    Delta = W_new - W_ref

che minimizza  MSE(W_ref + Delta, x, f)  soggetto a:

  Vincolo 1 - positività:
      W_ref[n] + Delta[n] >= 0   per ogni n

  Vincolo 2 - misura unitaria:
      sum_{n=0}^N (W_ref[n] + Delta[n]) = N+1
      Poiché sum(W_ref) = N+1, questo equivale a:
          sum(Delta) = 0
      (sum(Delta) = 0 NON è un vincolo indipendente: è una conseguenza
      del vincolo di misura unitaria applicato sia a W_ref che a W_new.)

  Vincolo 3-upper (opzionale) - BP stocasticamente MAGGIORE del riferimento:
      ∀h = 0,...,N-1:   sum_{n=0}^h Delta[n] <= 0
      ovvero:          cumsum(Delta)[h] <= 0
      equivale a:      cumsum(W_new)[h] <= cumsum(W_ref)[h]
      → sposta massa verso destra -> CDF_BP(x) ≥ CDF_ref(x) -> BP >=_st ref

  Vincolo 3-lower (opzionale) - BP stocasticamente MINORE del riferimento:
      ∀h = 0,...,N-1:   sum_{n=0}^{h} Delta[n] >= 0
      ovvero:          cumsum(Delta)[h] >= 0
      equivale a:      cumsum(W_new)[h] >= cumsum(W_ref)[h]
      → sposta massa verso sinistra -> CDF_BP(x) ≤ CDF_ref(x) -> BP <=_st ref

      (Nota: il vincolo sum_{n=h}^N Delta[n] <= 0 del lower è equivalente a
      cumsum(Delta)[h-1] >= 0 tramite sum(Delta)=0.)

Quindi:
  upper: cumsum(Delta)[h] <= 0 per ogni h  ->  F_BP(x) - F_ref(x) <= 0
  lower: cumsum(Delta)[h] >= 0 per ogni h  ->  F_BP(x) - F_ref(x) >= 0
"""

import numpy as np


def cdf_nodes_ref(N: int, x: np.ndarray, f: np.ndarray, dx: float) -> np.ndarray:
    """
    Soglia SO basata sulla CDF esatta di f valutata ai nodi k/N.

    Restituisce cs_ref[h] = (N+1) * F_f(h/N)  per h = 0..N,
    che rappresenta cumsum(W_ideal)[h] per una distribuzione che
    approssima f perfettamente.

    Usare questa soglia come riferimento SO significa imporre:
        upper: cumsum(W_new)[h] <= (N+1)*F_f(h/N)  -> BP >=_st f
        lower: cumsum(W_new)[h] >= (N+1)*F_f(h/N)  -> BP <=_st f
    """
    nodes   = np.linspace(0, 1, N + 1)
    cdf_f   = np.cumsum(np.maximum(f, 0)) * dx
    cdf_f  /= cdf_f[-1]
    return (N + 1) * np.interp(nodes, x, cdf_f)


def build_scipy_constraints(N: int, direction: str | None,
                             cs_ref: np.ndarray | None = None) -> list:
    """
    Vincoli per scipy.optimize.minimize nel formalismo Delta.

    La variabile di ottimizzazione è Delta = W_new - W_ref.

    Vincoli mandatori:
      - sum(Delta) = 0   (misura unitaria: sum(W_new)=N+1 e sum(W_ref)=N+1)

    Vincoli SO - simmetria upper/lower: entrambi sono "coda direzionale ≤ 0":

      upper:  cumsum sinistra -> destra ≤ 0
              sum(Delta[0:h+1]) ≤ thr[h]   per h = 0..N-1
              Se cs_ref è None: thr = 0  ->  SO relativo a W_ref

      lower:  cumsum destra -> sinistra ≤ 0
              sum(Delta[h:]) ≤ 0         per h = 1..N
              (sempre soglia 0: SO relativo a W_ref; equivalente a
               cumsum l->r ≥ 0 dato sum(Delta)=0, ma più coerente per simmetria)

    Nota: cs_ref è usato solo per upper (soglie opzionali); lower usa sempre 0.
    """
    # sum(Delta) = 0
    constraints = [{
        'type': 'eq',
        'fun': lambda Delta: np.sum(Delta),
        'jac': lambda Delta: np.ones(N + 1)
    }]

    # Soglia per upper: cs_ref se fornito, altrimenti 0 (SO relativo a W_ref)
    thresholds = cs_ref if cs_ref is not None else np.zeros(N)

    if direction == 'upper':
        # cumsum sinistra -> destra ≤ 0:  sum(Delta[0:h+1]) ≤ thr[h]  per h=0..N-1
        for h in range(N):
            e = np.zeros(N + 1); e[:h + 1] = 1.0
            constraints.append({
                'type': 'ineq',
                'fun': lambda Delta, h=h, thr=thresholds[h]:
                    thr - np.sum(Delta[:h + 1]),
                'jac': lambda Delta, e=e.copy(): -e,
            })
    elif direction == 'lower':
        # cumsum destra -> sinistra ≤ 0:  sum(Delta[h:]) ≤ 0  per h=1..N
        # (equivalente a cumsum l->r ≥ 0 dato sum(Delta)=0, ma più coerente
        #  con la simmetria upper/lower: entrambe sono "coda ≤ 0")
        for h in range(1, N + 1):
            e = np.zeros(N + 1); e[h:] = 1.0
            constraints.append({
                'type': 'ineq',
                'fun': lambda Delta, h=h: -np.sum(Delta[h:]),   # ≥0 -> sum(Delta[h:])≤0
                'jac': lambda Delta, e=e.copy(): -e,
            })

    return constraints


def stochastic_penalty_torch(W, W_ref_t, direction: str):
    """
    Penalità differenziabile PyTorch per violazioni SO.

    Penalizza: max(0, cumsum(Delta)[h])^2  per upper
               max(0, -cumsum(Delta)[h])^2 per lower

    dove Delta = W - W_ref_t.
    """
    import torch
    Delta = W - W_ref_t
    if direction == 'upper':
        # cumsum l->r ≤ 0
        cs = torch.cumsum(Delta, dim=0)[:-1]
        violations = torch.clamp(cs, min=0.0)
    else:
        # cumsum r->l ≤ 0:  sum(Delta[h:]) ≤ 0  per h=1..N
        # flip -> cumsum -> flip-back dà i tail-sum; poi si prende da h=1
        cs_right = torch.flip(
            torch.cumsum(torch.flip(Delta, [0]), dim=0), [0]
        )[1:]   # shape (N,)
        violations = torch.clamp(cs_right, min=0.0)
    return torch.sum(violations ** 2)


def check_order(W_new: np.ndarray, W_ref: np.ndarray,
                direction: str,
                cdf_ref: np.ndarray | None = None) -> dict:
    """
    Verifica numerica del vincolo SO.

    Se cdf_ref è fornito (shape N+1): controlla cumsum(W_new)[h] vs cdf_ref[h].
    Altrimenti controlla cumsum(Delta)[h] vs 0 (Delta = W_new - W_ref).
    """
    if cdf_ref is not None:
        cs_new = np.cumsum(W_new)[:len(cdf_ref)]
        if direction == 'upper':
            violations = np.maximum(cs_new - cdf_ref, 0.0)
        else:
            violations = np.maximum(cdf_ref - cs_new, 0.0)
    else:
        Delta    = W_new - W_ref
        if direction == 'upper':
            # cumsum l->r ≤ 0:  sum(Delta[0:h+1]) ≤ 0  per h=0..N-1
            cs = np.cumsum(Delta)[:-1]
            violations = np.maximum(cs, 0.0)
        else:
            # cumsum r->l ≤ 0:  sum(Delta[h:]) ≤ 0  per h=1..N
            cs_right = np.array([np.sum(Delta[h:])
                                  for h in range(1, len(Delta))])
            violations = np.maximum(cs_right, 0.0)

    return {
        'satisfied': float(violations.max()) < 1e-4,
        'max_violation': float(violations.max()),
    }


def compute_delta(W_new: np.ndarray, W_ref: np.ndarray) -> np.ndarray:
    """
    Delta = W_new - W_ref.

    Rappresenta lo spostamento di massa rispetto ai pesi di riferimento.
    sum(Delta) = 0 per costruzione (entrambi sommano a N+1).
    Delta[k] > 0 -> massa aggiunta alla k-esima base.
    Delta[k] < 0 -> massa sottratta dalla k-esima base.
    """
    return W_new - W_ref

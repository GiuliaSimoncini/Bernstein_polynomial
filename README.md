# Approssimazione con Polinomi di Bernstein
### Teoria, Metodi e Ordinamento Stocastico

---

## 1. La Base di Bernstein

I **polinomi di Bernstein di grado N** sono una famiglia di N+1 funzioni definite su $[0,1]$:

$$B_{k,N}(x) = \binom{N}{k} x^k (1-x)^{N-k} \quad k = 0, 1, \dots, N$$

Ogni base integra esattamente a $1/(N+1)$ su $[0,1]$, quindi per ottenere una **misura unitaria** con i pesi $W$ è sufficiente imporre $\sum W = N+1$.

Un **Bernstein Polynomial (BP)** con pesi $W$ è:

$$BP(W, x) = \sum_{k=0}^N W_k \cdot B_{k,N}(x)$$

Proprietà chiave:
- **Positività**: se $W_k \ge 0$ allora $BP(W,x) \ge 0$
- **Misura unitaria**: $\int_0^1 BP(W,x) \, dx = \frac{\sum_k W_k}{N+1}$, quindi il vincolo PDF si riduce a $\sum W = N+1$
- **Localizzazione**: la moda di $B_{k,N}$ cade in $k/N$, ogni base presidia la propria zona
- **Equivalenza con Taylor**: basi di Bernstein e Taylor di grado N generano lo stesso spazio, quindi ogni polinomio di grado $\le N$ ha una rappresentazione esatta in base di Bernstein

---

## 2. Il Problema di Ottimizzazione con Formulazione Delta

### L'operatore di Bernstein classico

L'**operatore di Bernstein classico** costruisce i pesi campionando direttamente la funzione target ai nodi $k/N$:

$$W_{\text{init}}[k] = f(k/N)$$

poi riscalati a $\sum W_{\text{init}} = N+1$. Questo dà un'approssimante ragionevole ma non ottima (in senso MSE). È il punto di partenza per tutti i metodi di ottimizzazione.

### La variabile di ottimizzazione: Delta

Dato un punto di riferimento $W_{\text{ref}}$ (l'operatore di Bernstein o il risultato dell'ottimizzazione non vincolata), introduciamo:

$$\Delta = W_{\text{new}} - W_{\text{ref}}$$

Il problema è: **trovare $\Delta$ che minimizza $MSE(W_{\text{ref}} + \Delta, x, f)$**

soggetto ai **vincoli mandatori**:
1. $W_{\text{ref}}[n] + \Delta[n] \ge 0$ per ogni $n$ (positività dei pesi)
2. $\sum_{n=0}^N (W_{\text{ref}}[n] + \Delta[n]) = N+1$ (misura unitaria)

**Nota importante:** il vincolo 2 implica automaticamente $\sum \Delta = 0$. Questo NON è un vincolo aggiuntivo indipendente: è una *conseguenza* del fatto che sia $W_{\text{ref}}$ che $W_{\text{new}}$ devono soddisfare la misura unitaria. Quindi:

$$\sum W_{\text{new}} = N+1 \quad \text{e} \quad \sum W_{\text{ref}} = N+1 \implies \sum \Delta = 0$$

---

## 3. Ordinamento Stocastico (SO)

### Cos'è e perché si usa la CDF

L'**ordinamento stocastico del 1° ordine** è una relazione tra distribuzioni.
Si dice che $X$ domina stocasticamente $Y$ ($X \ge_{st} Y$) se:

$$F_X(x) \le F_Y(x) \quad \forall x$$

**Perché si usa la CDF e non la PDF?** Perché la dominanza stocastica è un concetto intrinsecamente cumulativo: governa probabilità di superare soglie, quantili e valor medio. Due densità possono incrociarsi liberamente eppure produrre un ordinamento netto sulle CDF.

### Vincoli opzionali con Formulazione Delta

I vincoli SO si traducono in **vincoli lineari su $\text{cumsum}(\Delta)$**, dove $\Delta = W_{\text{new}} - W_{\text{ref}}$:

#### Vincolo 3-upper - BP stocasticamente MAGGIORE del riferimento

$$\sum_{n=0}^h \Delta[n] \le 0 \quad \implies \quad \text{cumsum}(\Delta)[h] \le 0 \quad \forall h = 0, \dots, N-1$$

Equivale a: $\text{cumsum}(W_{\text{new}})[h] \le \text{cumsum}(W_{\text{ref}})[h]$ ossia i pesi si spostano verso destra. La CDF del BP sta **sotto** quella di riferimento, quindi il BP ha **più massa a destra**: $X_{BP} \ge_{st} X_{\text{ref}}$.

#### Vincolo 3-lower - BP stocasticamente MINORE del riferimento

$$\sum_{n=h}^N \Delta[n] \le 0 \quad \iff \quad \text{cumsum}(\Delta)[h] \ge 0 \quad \forall h = 0, \dots, N-1$$

Equivale a: $\text{cumsum}(W_{\text{new}})[h] \ge \text{cumsum}(W_{\text{ref}})[h]$ ossia i pesi si spostano verso sinistra. La CDF del BP sta **sopra** quella di riferimento, quindi il BP ha **più massa a sinistra**: $X_{BP} \le_{st} X_{\text{ref}}$.

### A quale riferimento si applica il SO?

Il sistema implementa SO rispetto a **due diversi punti di riferimento**, a seconda del metodo:

| Metodo | $W_{\text{ref}}$ | Significato |
|---|---|---|
| `bernstein_op_upper/lower` | $W_{\text{bernsteinOp}}$ | SO rispetto all'operatore di Bernstein classico |
| `scipy_upper/lower` | $W_{\text{scipy}}$ | SO rispetto all'ottimo non vincolato (MSE minimo) |
| `pytorch_upper` | $W_{\text{pytorch}}$ | idem, con vincolo soft (penalità) |

---

## 4. I Metodi a Confronto

### A. Ing-Cane - soluzione quasi-chiusa

Attiva solo le due basi di Bernstein che racchiudono la moda di $f$, distribuendo la massa (N+1) in proporzione inversa alla distanza dalla moda.

**Vantaggi**: costo $\mathcal{O}(N)$, $< 0.1$ ms. È la versione elementare dell'idea di pre-emphasis locale: si enfatizzano solo le basi vicine al picco.

**Svantaggi**: con 2 basi attive su N+1 ignora la forma globale di $f$. MSE molto alto. Non supporta vincoli SO.

---

### B. Scipy SLSQP - ottimizzazione vincolata esatta

Minimizza $MSE(W_{\text{new}})$ con *Sequential Least Squares Programming*, usando il gradiente analitico:

$$\nabla_W \text{MSE}(W) = -\frac{2}{n} M^T (f - MW)$$

I vincoli sono gestiti **esattamente** (non come penalità):
- bounds per la positività
- equality constraint per $\sum W_{\text{new}} = N+1$
- $N$ disuguaglianze lineari per il SO

**Versione non vincolata** (`scipy`): trova l'ottimo globale del problema convesso. $\Delta = 0$ per costruzione ($W_{\text{ref}} = W_{\text{init}}$ = warm-start).

**Versione SO** (`scipy_upper/lower`): $W_{\text{ref}} = W_{\text{scipy}}$ (l'ottimo non vincolato). I vincoli $\text{cumsum}(\Delta) \le 0$ / $\ge 0$ rimpiccioliscono il feasible set ma mantengono la convessità del problema. SLSQP garantisce l'ottimo locale nel feasible set SO che per questo problema è globale.

**Versione SO da BernOp** (`bernstein_op_upper/lower`): $W_{\text{ref}} = W_{\text{bernsteinOp}}$. Il vincolo SO è ora rispetto al punto di partenza dell'operatore classico, non rispetto all'ottimo. Produce un BP che è vincolato a stare stocasticamente sopra/sotto l'operatore di Bernstein.

---

### C. PyTorch - gradient descent non vincolato

Reparametrizza i pesi tramite softmax per garantire per costruzione $W_k > 0$ e $\sum W = N+1$:

$$W = \text{softmax}(W_{\text{raw}}) \cdot (N+1)$$

Introduce una reparametrizzazione non lineare che rende il paesaggio non convesso in $W_{\text{raw}}$. In pratica raggiunge MSE paragonabile a SLSQP ma impiegando 100 - 1000x più tempo. Utile per validazione o per integrazione in pipeline di deep learning.

---

### D. PyTorch con penalità SO

Aggiunge alla loss un termine per le violazioni del vincolo SO:

$$\mathcal{L}(W) = \text{MSE}(W) + \lambda \sum_h \max(0, \text{cumsum}(\Delta)[h] \cdot \text{segno})^2$$

dove $\text{segno} = +1$ per upper (viola se $\text{cumsum}(\Delta) > 0$) e $\text{segno} = -1$ per lower. $W_{\text{ref}} = W_{\text{pytorch}}$.

**Differenza fondamentale rispetto a Scipy+SO**: la penalità è un'approssimazione *soft*, nessun valore di $\lambda$ garantisce il vincolo esatto. Scipy SLSQP impone invece un vincolo *hard*. Nei nostri esperimenti, PyTorch+penalità può ancora violare il vincolo SO (SO:✗) anche con $\lambda=200$, mentre Scipy+SO è sempre ✓.

---

## 5. Come Funziona SLSQP Internamente

SLSQP (*Sequential Least Squares Programming*) è un algoritmo iterativo per problemi della forma: minimizzare $f(x)$ soggetto a:
- $g_i(x) = 0$ (uguaglianze)
- $h_j(x) \ge 0$ (disuguaglianze)
- $lb \le x \le ub$ (bounds)

Ad ogni iterazione esegue due passi.

**Passo 1 - sottoproblema QP locale.**
Approssima $f(x)$ con una forma quadratica usando il gradiente e una stima dell'Hessiana (aggiornamento BFGS) e linearizza i vincoli attorno al punto corrente $x_k$. Risolve questo *Quadratic Program* (QP) in modo esatto, ottenendo una direzione di discesa $d_k$.

**Passo 2 - line search.**
Cerca lungo la direzione $d_k$ il passo $\alpha$ che riduce una funzione di merito. Aggiorna $x_{k+1} = x_k + \alpha \cdot d_k$. Ripete finché il gradiente proiettato è abbastanza piccolo.

### Perché è lento con N grande

Il QP locale ha N+1 variabili e circa N vincoli di disuguaglianza SO. Ad ogni iterazione SLSQP deve:
- Aggiornare l'Hessiana approssimata: $\mathcal{O}(N^2)$ memoria e tempo
- Risolvere il QP vincolato: $\mathcal{O}(N^3)$ nel caso peggiore

Con $N=30$ il costo è gestibile ($< 1 s$). Con $N=100$ il QP interno è circa 30x più costoso e servono più iterazioni perché lo spazio feasibile con 99 vincoli SO è molto più rigido.

### Gradiente analitico

Il codice fornisce il gradiente esplicitamente a SLSQP. Poiché $\Delta$ appare linearmente nell'obiettivo, il gradiente rispetto a $\Delta$ coincide con quello rispetto a $W$:

$$\nabla_\Delta \text{MSE}(W_{\text{ref}} + \Delta) = -\frac{2}{n} M^T (f - M(W_{\text{ref}} + \Delta))$$

Questo accelera la convergenza di circa 2-5x rispetto all'uso del gradiente numerico.

---

## 6. Effetto del Grado N

N è il **grado del polinomio di Bernstein**. Aumentare N significa simultaneamente: più basi (N+1), più pesi da ottimizzare e polinomio approssimante di grado più alto.

### Effetto teorico

Ogni base $B_{k,N}$ ha la sua moda in $k/N$ e larghezza proporzionale a $1/\sqrt{N}$. Con N grande le basi diventano più strette e più fitte. Per il **teorema di Weierstrass in forma di Bernstein**, l'operatore converge uniformemente a $f$ quando $N \to \infty$, con velocità $\mathcal{O}(1/N)$.

### Effetto per ciascun metodo

| Metodo | N piccolo | N grande |
|---|---|---|
| **Bernstein op** | Forte smoothing, errore alto | Converge lentamente a $f$ con $\mathcal{O}(1/N)$ |
| **Scipy** | Veloce, limitato dal modello | Preciso (MSE $\to 0$), ma lento con SO ($\mathcal{O}(N^3)$) |
| **PyTorch** | Simile a Scipy | Serve più epochs e $\lambda \approx 50N$ |

---

## 7. Chiarimento: bernstein_op_upper/lower non è Bernstein con SO

Questo è un punto concettuale importante. `bernstein_op_upper` e `bernstein_op_lower` **non** sono metodi in cui l'operatore di Bernstein risolve il vincolo SO. L'ottimizzazione è sempre eseguita da **SLSQP**. La parola Bernstein nel nome indica solo che il **riferimento SO** è l'operatore di Bernstein.

| Metodo | Chi calcola $W_{\text{ref}}$ | Chi ottimizza $W_{\text{new}}$ |
|---|---|---|
| `bernstein_op` | Operatore di Bernstein | nessuno (formula chiusa) |
| `bernstein_op_upper/lower` | Operatore di Bernstein | SLSQP vincolato |
| `scipy` | - | SLSQP non vincolato |
| `scipy_upper/lower` | SLSQP non vincolato | SLSQP vincolato |
| `pytorch` | - | Gradient descent (Adam) |
| `pytorch_upper` | Grandient descent (Adam) | Gradient descent con penalità |

In pratica `bernstein_op_upper` risponde alla domanda: *"Qual è il BP che minimizza l'MSE con il vincolo di essere stocasticamente maggiore dell'operatore di Bernstein?"*. Per questo motivo hanno quasi sempre MSE **inferiore** a `bernstein_op`: SLSQP ottimizza attivamente a partire da quel punto.

---

## 8. Cosa Viene Stampato

Per ogni metodo SO il codice stampa:
- Pesi $W$ ottimizzati ($\sum W = N+1$)
- $\Delta$ calcolato come $W_{\text{new}} - W_{\text{ref}}$ ($\sum \Delta = 0$)
- Verifica numerica per $\sum \Delta \approx 0$
- Stato del vincolo SO esaminando i limiti min e max di $\text{cumsum}(\Delta)$

---

## 9. Il Corridoio Stocastico

Nel grafico `expN_so_summary.png` compaiono **due corridoi**:
- **Verde** (scipy): tra CDF di `scipy_upper` e `scipy_lower` - il range di CDF ottenibili minimizzando l'MSE rispetto all'ottimo $W_{\text{scipy}}$
- **Arancione** (BernOp): tra CDF di `bernstein_op_upper` e `bernstein_op_lower` - il range di CDF ottenibili con SO rispetto all'operatore di Bernstein

---

## 10. Struttura del Codice

La struttura del progetto è la seguente:
* `main.py` ← orchestrazione: esperimenti + grafici
* `bernstein/basis.py` ← base $B_{k,N}$, eval_bp, MSE, warm-start
* `bernstein/stochastic.py` ← vincoli SO e verifica nel formalismo $\Delta$
* `bernstein/methods.py` ← solutori (Ing-Cane, Scipy, PyTorch)
* `bernstein/experiments.py` ← runner con log di $W$ e $\Delta$
* `bernstein/plotting.py` ← funzioni di visualizzazione

### Grafici prodotti (per ogni esperimento N)

| File | Contenuto |
|---|---|
| `expN_pdf.png` | PDF di tutti i metodi vs $f(x)$ + barplot MSE con tempi |
| `expN_cdf.png` | CDF di tutti i metodi vs CDF target |
| `expN_weights.png` | Istogramma pesi $W$ per metodo |
| `expN_delta.png` | Istogramma $\Delta$ + $\text{cumsum}(\Delta)$ per metodi SO |
| `expN_so_upper.png` / `lower.png`| SO Scipy: CDF + zoom zona critica |
| `expN_bo_so_upper.png` / `lower.png`| SO BernOp: CDF + zoom |
| `expN_so_summary.png` | Tutte le CDF SO + corridoi upper/lower |

---

## 11. File di Esecuzione (Entry Points)

Sono stati realizzati 4 tipi di file `main.py` per l'esecuzione di esperimenti diversi:

| File | Scopo dell'Esperimento | Metodi Confrontati |
|---|---|---|
| `completo.py` | Confronta tutti i metodi di approssimazione implementati. | Ing-cane, BernOp (base, SO upper, SO lower), Scipy SLSQP (base, SO upper, SO lower), PyTorch (base, SO). |
| `bern_scipy_torch_noSO.py` | Confronta solo i 3 metodi con il miglior MSE, escludendo l'ordinamento stocastico. | Bernstein Operator, Scipy SLSQP, PyTorch. |
| `bern_bernSOupp_bernSOlow.py` | Confronta esclusivamente l'approssimazione del Bernstein Operator, con e senza SO. | Bernstein Operator (base, SO upper, SO lower). |
| `bern_base_alternativa_n_grande.py`| Confronta il Bernstein Operator senza SO, ma con un N molto grande. Usa un calcolo alternativo per la base. | Bernstein Operator. |

---

## 12. Funzioni Target Analizzate

Le prestazioni dei vari metodi sono testate su diversi tipi di funzioni target $f$, tra cui:
- **Unimodale Beta**: utile per testare il comportamento su una singola moda.
- **Polinomiale**: funzione liscia di test.
- **Bimodale**: ottenuta come somma di due funzioni Beta, ideale per testare la risposta a picchi multipli.

---

## 13. Grafici Generati

Dall'esecuzione del codice vengono realizzati numerosi grafici di confronto, tra cui:
- **Grafico delle PDF / CDF**: illustrano le densità e ripartizioni stimate rispetto a quella target.
- **Grafico dei pesi W / Delta**: istogrammi che mostrano la distribuzione dei parametri ottimizzati.
- **Grafico dei Tempi e dell'MSE**: diagramma a barre che illustra l'efficienza algoritmica e l'errore.

---
---

## APPENDICE - Normalizzazione della Base

Nel notebook Mathematica viene usata la base standard di Bernstein non normalizzata:

```mathematica
T[x_, n1_, nn_] := Binomial[nn, n1] * x^n1 * (1-x)^(nn-n1)
```

che corrisponde esattamente a:

$$B_{k,N}(x) = \binom{N}{k} x^k (1-x)^{N-k}$$

Ogni $B_{k,N}$ integra a $1/(N+1)$, quindi la somma dei pesi $W$ per avere misura unitaria deve essere N+1. La convenzione alternativa (base normalizzata) usa:

$$M_{k,N}(x) = (N+1) \cdot B_{k,N}(x)$$

dove ogni funzione integra a 1, e quindi $\sum W = 1$. Le due convenzioni sono matematicamente equivalenti: cambia solo la scala numerica dei pesi, mentre la logica di $\Delta$, SO e ottimizzazione rimane identica.

### Perché la somma dei pesi è N+1 (e non 1)?

Quando si approssima una funzione di densità di probabilità (PDF), è un vincolo stringente che l'area totale sotto la curva sia pari a 1. Se imponessimo ingenuamente che i pesi sommino a 1 ($\sum W = 1$), la nostra approssimante finirebbe per avere un'area totale errata, invalidandola come distribuzione di probabilità. 

Il motivo geometrico e analitico risiede in una proprietà fondamentale della base di Bernstein standard: **le singole funzioni di base non hanno un'area unitaria**. 

Se calcoliamo l'integrale definito su [0,1] di una singola base di Bernstein $B_{k,N}(x)$, otteniamo un valore strettamente dipendente dal grado N:

$$\int_0^1 B_{k,N}(x) \, dx = \frac{1}{N+1}$$

Quando costruiamo il nostro polinomio approssimante $BP(W, x)$ come combinazione lineare di queste basi, l'integrale dell'intera funzione diventa la somma degli integrali delle singole basi, moltiplicati per i rispettivi pesi $W_k$:

$$\int_0^1 BP(W,x) \, dx = \sum_{k=0}^N W_k \left( \int_0^1 B_{k,N}(x) \, dx \right) = \frac{\sum_{k=0}^N W_k}{N+1}$$

Affinché il nostro polinomio finale mantenga la **misura unitaria** (cioè rappresenti una vera PDF con area 1), dobbiamo imporre che l'integrale totale sia esattamente 1:

$$\frac{\sum_{k=0}^N W_k}{N+1} = 1 \quad \implies \quad \sum_{k=0}^N W_k = N+1$$

In sintesi: poiché ogni base di Bernstein contribuisce solo per un'area pari a $1/(N+1)$, la somma totale dei pesi deve essere scalata a $N+1$ per compensare questo fattore e garantire un'area finale unitaria.


### Stabilità Numerica ed Efficienza della Base di Bernstein

Il calcolo della base di Bernstein $B_{k,N}(x) = \binom{N}{k} x^k (1-x)^{N-k}$ nasconde un'insidia computazionale notevole quando il grado N cresce. Sostituire l'implementazione iterativa classica (`basis_matrix`) con la versione basata sulla distribuzione binomiale (`basis_matrix_stable`) è fondamentale per due motivi principali:

* **Stabilità Numerica (Prevenzione di Overflow/Underflow)**: Nell'implementazione base, i tre termini $\binom{N}{k}$, $x^k$ e $(1-x)^{N-k}$ vengono calcolati separatamente e poi moltiplicati. Per N grande (già intorno a N=60 o N=100), il coefficiente binomiale esplode a valori enormi superando i limiti della precisione di macchina (*overflow*), mentre i termini esponenziali diventano infinitesimi (*underflow*). Il risultato di questa moltiplicazione estrema porta a gravi errori di arrotondamento o a valori `NaN`. La versione `_stable` risolve il problema sfruttando un'elegante equivalenza matematica: la base di Bernstein coincide esattamente con la **Probability Mass Function (PMF)** della distribuzione Binomiale. Scipy calcola la PMF lavorando internamente nello spazio dei logaritmi (tramite funzioni come `gammaln`), garantendo una precisione perfetta senza mai generare i numeri giganteschi intermedi.
* **Efficienza Vettoriale (Broadcasting)**: La funzione originale utilizza un ciclo `for` per iterare sui gradi k, un'operazione notoriamente lenta in Python. La versione `_stable` sfrutta invece il *broadcasting* multidimensionale di NumPy tramite `np.newaxis`. Questo permette di valutare l'intera matrice NxM in una singola operazione vettorizzata eseguita in C, abbattendo drasticamente i tempi di esecuzione, specialmente negli scenari in cui la griglia o il grado N sono molto densi.
"""
Analisi statistica dei conteggi radioattivi
============================================
Struttura del file
------------------
  PARTE 1 — CALCOLI
    1.1  Binning dei timestamp in finestre di ampiezza Δt
    1.2  Stima di λ̂ e del tasso r per ogni misura
    1.3  Miglior stima del tasso (media pesata, BLUE)
    1.4  Frazioni di "0 conteggi" e "0 o 1 conteggio"
    1.5  Posterior pdf di r (Gamma con prior uniforme)
    1.6  Tempi di attesa (Erlang)
    1.7  Pallinometro — inferenza su p e test chi²

  PARTE 2 — GRAFICI
    2.1  Istogrammi conteggi per finestra (uno per Δt)
    2.2  Probabilità di 0 conteggi vs Δt
    2.3  Probabilità di 0 o 1 conteggio vs Δt
    2.4  Posterior pdf di r — tre casi
    2.5  Tempi di attesa — confronto con Erlang
    2.6  Pallinometro — distribuzioni e posterior di p
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, gamma as gamma_dist, erlang, beta as beta_dist, binom
from scipy.stats import chi2 as chi2_dist
from scipy.optimize import brentq
import lab_tools as lt


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURAZIONE GLOBALE
# ═════════════════════════════════════════════════════════════════════════════

# Ampiezza delle finestre temporali usate nelle cinque acquisizioni [s]
dt = [1, 2, 3, 5, 10]

# Conteggi osservati per le due configurazioni del pallinometro,
# letti direttamente dai file di acquisizione.
#
# Formato atteso dei file: una colonna, un valore per riga,
# ogni valore = numero di palline finite in quel bin (in ordine da sinistra a destra).
# Esempio con 9 bin:
#   5          <- bin 0 (piu' a sinistra)
#   18
#   ...
#   2          <- bin 8 (piu' a destra)
#
# Se il file ha due colonne (es. "bin  conteggio"), cambiare n_cols=1 in n_cols=2
# e tenere [0] per la prima colonna oppure [1] per la seconda.
CONTEGGI11 = lt.read_multiple_variables("pallinometroNormale1.txt", n_cols=2)[1]
CONTEGGI12 = lt.read_multiple_variables("pallinometroNormale2.txt", n_cols=2)[1]

CONTEGGI_CASO1 = []

for i, j in zip(CONTEGGI11, CONTEGGI12):
    CONTEGGI_CASO1.append(i + j)

CONTEGGI21 = lt.read_multiple_variables("pallinometroStorto1.txt", n_cols=2)[1]
CONTEGGI22 = lt.read_multiple_variables("pallinometroStorto2.txt", n_cols=2)[1]

CONTEGGI_CASO2 = []

for i, j in zip(CONTEGGI21, CONTEGGI22):
    CONTEGGI_CASO2.append(i + j)
    
CONTEGGI_CASO1 = np.array(CONTEGGI_CASO1)
CONTEGGI_CASO2 = np.array(CONTEGGI_CASO2)

# N_BINS e N_URTI si ricavano direttamente dai dati: non serve cambiarli a mano
# anche se i due strumenti hanno un numero di bin diverso.
# (I due file devono avere lo stesso numero di righe, cioe' lo stesso pallinometro.)
N_BINS = len(CONTEGGI_CASO1)
N_URTI = N_BINS - 1   # numero di file di chiodi = numero di scelte binarie per pallina

RNG = np.random.default_rng(seed=42)   # generatore riproducibile per la simulazione


# ═════════════════════════════════════════════════════════════════════════════
#  PARTE 1 — CALCOLI
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  1.1 — Lettura dei file e binning manuale dei timestamp
#
#  Per ogni acquisizione i si legge un file con due colonne:
#    colonna 0 = conteggio grezzo
#    colonna 1 = timestamp dell'evento [s]
#
#  I timestamp vengono distribuiti in finestre consecutive di ampiezza Δt:
#    finestra j → [j·Δt, (j+1)·Δt)
#    val[j] = numero di eventi caduti in quella finestra
# ─────────────────────────────────────────────────────────────────────────────

all_vals      = []   # all_vals[i]  = vettore dei conteggi per finestra (misura i)
all_n_windows = []   # all_n_windows[i] = numero di finestre della misura i

for i in range(1, 6):
    passo = dt[i - 1]

    count, times = lt.read_multiple_variables(f"lettura{i}.txt", n_cols=2)
    times = np.array(times)

    # binning: scorre i timestamp e conta gli eventi in ogni finestra
    val   = []
    start = 0.0
    fine  = passo
    idx   = 0

    while idx < len(times):
        conteggio = 0
        while idx < len(times) and times[idx] < fine:
            if times[idx] >= start:
                conteggio += 1
            idx += 1
        val.append(conteggio)
        start = fine
        fine += passo

    if len(val) < 60:
        n = len(val)
        for j in range(60 - n):
            val.append(0)
            
    all_vals.append(np.array(val))
    all_n_windows.append(len(val))
    
    
# ─────────────────────────────────────────────────────────────────────────────
#  1.2 — Stima di λ̂ e del tasso r per ogni misura
#
#  Modello: i conteggi per finestra seguono una distribuzione di Poisson
#  con parametro λ = r · Δt, dove r è il tasso di eventi [s⁻¹].
#
#  Stimatore di massima verosimiglianza (MLE) per λ:
#    λ̂ = media campionaria dei conteggi per finestra
#
#  Incertezza (deviazione standard della stima):
#    σ(λ̂) = √(λ̂ / n)    dove n = numero di finestre
#
#  Conversione al tasso:
#    r = λ̂ / Δt
#    σ(r) = σ(λ̂) / Δt
# ─────────────────────────────────────────────────────────────────────────────

lambdas      = []
sigmaLambdas = []
rates        = []
sigmaRates   = []

for i, (val, passo) in enumerate(zip(all_vals, dt)):
    n    = len(val)
    lam  = np.mean(val)
    slam = np.sqrt(lam / n)

    lambdas.append(lam)
    sigmaLambdas.append(slam)
    rates.append(lam / passo)
    sigmaRates.append(slam / passo)
    
# ─────────────────────────────────────────────────────────────────────────────
#  1.3 — Miglior stima del tasso: media pesata inverso-varianza (BLUE)
#
#  Le cinque stime r_i sono indipendenti e misurano lo stesso parametro r.
#  La combinazione ottimale (a varianza minima) è la media pesata con:
#    
#    w_i = 1 / σ²(r_i):
#
#    r_best = Σ(w_i · r_i) / Σ(w_i)
#    σ(r_best) = 1 / √(Σ w_i)
# ─────────────────────────────────────────────────────────────────────────────

pesi       = np.array([1 / s**2 for s in sigmaRates])
r_best     = np.sum(pesi * np.array(rates)) / np.sum(pesi)
r_best_err = 1 / np.sqrt(np.sum(pesi))

# ── stampa riepilogo ─────────────────────────────────────────────────────────
print("─" * 60)
print(f"{'Δt':>4}  {'λ̂':>12}  {'σ(λ̂)':>12}  {'r [s⁻¹]':>12}  {'σ(r)':>12}")
print("─" * 60)
for passo, lam, slam, r, sr in zip(dt, lambdas, sigmaLambdas, rates, sigmaRates):
    print(f"{passo:>4}  {lam:>12.6f}  {slam:>12.6f}  {r:>12.6f}  {sr:>12.6f}")
print("─" * 60)
print(f"\nr_best = {r_best:.6f} ± {r_best_err:.6f} s⁻¹\n")

# ─────────────────────────────────────────────────────────────────────────────
#  1.4 — Frazioni di "0 conteggi" e "0 o 1 conteggio" per ogni Δt
#
#  Definizioni:
#    f0  = #{finestre con 0 eventi}      / n   →  stima di P(X = 0)
#    f01 = #{finestre con 0 o 1 evento}  / n   →  stima di P(X ≤ 1)
#
#  Incertezza (approssimazione gaussiana per proporzioni binomiali, valida
#  per n grande):
#    σ(f) = √( f · (1 − f) / n )
#
#  Valori attesi da una Poisson con tasso r_best:
#    P(X = 0)     = e^{−r·Δt}
#    P(X ≤ 1)     = (1 + r·Δt) · e^{−r·Δt}
# ─────────────────────────────────────────────────────────────────────────────

f0_obs  = []
f0_err  = []
f01_obs = []
f01_err = []

for val in all_vals:
    n   = len(val)
    f0  = np.mean(val == 0)
    f01 = np.mean(val <= 1)
    f0_obs.append(f0);   f0_err.append(np.sqrt(f0  * (1 - f0)  / n))
    f01_obs.append(f01); f01_err.append(np.sqrt(f01 * (1 - f01) / n))

# Curve teoriche su un asse Δt fine (nessuna incertezza, solo r_best)
t_fine      = np.linspace(0, 12, 500)
p0_teoria   = np.exp(-r_best * t_fine)
p01_teoria  = (1 + r_best * t_fine) * np.exp(-r_best * t_fine)

for i, j in zip(f0_obs, f0_err):
    print(f"{i:.6f} +/- {j:.6f}")
    

for i, j in zip(f01_obs, f01_err):
    print(f"{i:.6f} +/- {j:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
#  1.5 — Posterior pdf di r con prior uniforme (punto 7, opzionale)
#
#  Con prior p(r) ∝ 1 (uniforme su r ≥ 0), la posterior è proporzionale
#  alla likelihood di Poisson:
#
#    p(r | dati) ∝ r^{ΣX_j} · e^{−n·r·Δt}
#
#  Questa è una distribuzione Gamma:
#    r | dati  ~  Gamma(α, scale = 1/β)
#    α = ΣX_j + 1      (somma conteggi + 1 dal prior)
#    β = n · Δt        (tempo totale osservato)
#
#  Per dati combinati di più gruppi:
#    α_tot = Σ_i ΣX_{ij} + 1
#    β_tot = Σ_i n_i · Δt_i
# ─────────────────────────────────────────────────────────────────────────────

def calcola_posterior_r(indici, passi):
    """
    Calcola le statistiche della Gamma posterior per il tasso r.

    Parametri
    ----------
    indici : lista di indici nelle liste all_vals / dt da combinare
    passi  : lista dei corrispondenti Δt

    Restituisce
    -----------
    dizionario con: alpha, beta, media, mediana, std, intervallo_683, dist
    """
    alpha = sum(all_vals[k].sum() for k in indici) + 1
    beta  = sum(len(all_vals[k]) * passi[j] for j, k in enumerate(indici))
    dist  = gamma_dist(a=alpha, scale=1 / beta)

    media   = dist.mean()    # E[r] = α / β
    mediana = dist.median()
    std     = dist.std()

    # Intervallo simmetrico intorno alla media che contiene il 68.3% della probabilità:
    # si cerca numericamente δ tale che CDF(μ+δ) − CDF(μ−δ) = 0.683
    def prob_intorno_media(delta):
        lo = max(0.0, media - delta)
        return dist.cdf(media + delta) - dist.cdf(lo) - 0.683

    delta      = brentq(prob_intorno_media, 1e-9, 10 * std)
    intervallo = (max(0.0, media - delta), media + delta)

    return dict(alpha=alpha, beta=beta,
                media=media, mediana=mediana, std=std,
                intervallo_683=intervallo, dist=dist)

# Tre casi: solo Δt=1s, solo Δt=5s, tutte le misure combinate
casi_posterior = {
    "Δt = 1 s": calcola_posterior_r([0],          [dt[0]]),
    "Δt = 5 s": calcola_posterior_r([3],          [dt[3]]),
    "Tutte":    calcola_posterior_r(list(range(5)), dt),
}

# Stampa riepilogo posterior
print("\n" + "─" * 75)
print(f"{'Caso':<14} {'Media':>10} {'Mediana':>10} {'Std':>10} "
      f"{'Int. 68.3% inf':>16} {'Int. 68.3% sup':>16}")
print("─" * 75)
for nome, res in casi_posterior.items():
    lo, hi = res["intervallo_683"]
    print(f"{nome:<14} {res['media']:>10.6f} {res['mediana']:>10.6f} "
          f"{res['std']:>10.6f} {lo:>16.6f} {hi:>16.6f}")
print("─" * 75)

# ─────────────────────────────────────────────────────────────────────────────
#  1.6 — Tempi di attesa (distribuzione di Erlang)
#
#  Dall'acquisizione a Δt = 10 s si estraggono i tempi di attesa per k eventi
#  consecutivi non sovrapposti:
#    τ_j = t[j·k + k − 1] − t[j·k]
#
#  La distribuzione attesa è una Erlang(k, r_best):
#    f(τ) = r^k · τ^{k−1} · e^{−r·τ} / (k−1)!
#  Per k = 1 coincide con un'esponenziale di parametro r_best.
#
#  La media teorica è E[τ] = k / r_best.
# ─────────────────────────────────────────────────────────────────────────────

_, times5 = lt.read_multiple_variables("lettura5.txt", n_cols=2)
times5 = np.array(times5)

k_list           = [1, 2, 3, 5]
tau_per_k        = {}   # tau_per_k[k] = array dei tempi di attesa osservati
statistiche_tau  = {}   # statistiche_tau[k] = dict con media osservata e teorica

for k in k_list:
    n_gruppi = (len(times5) - 1) // k
    tau = np.array([times5[j * k + k] - times5[j * k]
                    for j in range(n_gruppi)])

    # Incertezza sulla media (Errore Standard)
    # Usiamo il valore teorico per r_best come riferimento
    sigma_tau_media = np.sqrt(k) / (r_best * np.sqrt(n_gruppi))

    tau_per_k[k]       = tau
    statistiche_tau[k] = dict(n_gruppi     = n_gruppi,
                               media_obs    = tau.mean(),
                               media_teoria = k / r_best,
                               std_media    = sigma_tau_media)

    print(f"k={k}: n={n_gruppi:4d} | "
          f"τ̄_obs = {tau.mean():.4f} ± {sigma_tau_media:.4f} s | "
          f"τ̄_teo = {k / r_best:.4f} ± {k / r_best**2 * r_best_err} s")
    
for k in k_list:
    n_gruppi = (len(times5) - 1) // k
    tau = np.array([times5[j * k + k] - times5[j * k]
                    for j in range(n_gruppi)])

    # --- CALCOLO SPERIMENTALE ---
    media_obs = tau.mean()
    # Errore standard calcolato dai dati:
    sigma_media_obs = tau.std(ddof=1) / np.sqrt(n_gruppi)
    
    # --- CALCOLO TEORICO ---
    media_teo = k / r_best
    # Incertezza che il modello prevede per questo numero di misure:
    sigma_media_teo = (np.sqrt(k) / r_best) / np.sqrt(n_gruppi)

    print(f"k={k}:")
    print(f"  Osservata: {media_obs:.4f} ± {sigma_media_obs:.4f} s")
    print(f"  Attesa:    {media_teo:.4f} ± {sigma_media_teo:.4f} s")
    print(f"  Accordo:   {np.abs(media_obs - media_teo) / sigma_media_obs:.2f} sigma")
    
# ─────────────────────────────────────────────────────────────────────────────
#  1.7 — Scintillatore: separazione del rate cosmico dal rate ambientale
#         e misura del contributo radioattivo del tufo
#
#  Tre acquisizioni a Δt = 10 s:
#    • lettura5: scintillatore orizzontale (aria)   → cattura 100% dei cosmici
#    • lettura6: scintillatore verticale   (aria)   → cattura  50% dei cosmici
#    • lettura7: scintillatore orizzontale + tufo   → come lettura5 + contributo tufo
#
#  Il rate misurato è la sovrapposizione di cosmici e fondo ambientale:
#    R_orizz = R_cosmici · 1.00 + R_amb · η
#    R_vert  = R_cosmici · 0.50 + R_amb · η
#  dove η = 0.04 è l'efficienza strumentale per la radiazione ambientale isotropa.
#
#  ── Separazione cosmici / ambientale ─────────────────────────────────────────
#  Sottraendo le due equazioni:
#    R_orizz − R_vert = R_cosmici · 0.50
#    R_cosmici = 2 · (R_orizz − R_vert)
#
#  Il contributo ambientale effettivo (già pesato per η):
#    R_amb · η = R_orizz − R_cosmici = 2·R_vert − R_orizz
#  Il rate ambientale fisico vero:
#    R_amb = (2·R_vert − R_orizz) / η
#
#  ── Contributo radioattivo del tufo ──────────────────────────────────────────
#  Con il tufo affiancato allo scintillatore orizzontale, il rate misurato è:
#    R_orizz_tufo = R_cosmici · 1.00 + R_amb · η + R_tufo · η
#  (i cosmici non cambiano; il tufo aggiunge radiazione ambientale isotropa)
#
#  Sottraendo la misura senza tufo:
#    R_orizz_tufo − R_orizz = R_tufo · η
#    R_tufo = (R_orizz_tufo − R_orizz) / η
# ─────────────────────────────────────────────────────────────────────────────
 
# Dimensioni fisiche dello scintillatore [m]
altezza_scint    = 0.01
larghezza_scint  = 0.05
lunghezza_scint  = 0.05
 
area_orizzontale = lunghezza_scint * larghezza_scint   # faccia grande [m²]
area_verticale   = lunghezza_scint * altezza_scint     # faccia piccola [m²]
 
ETA_AMBIENTALE = 0.04   # efficienza dello strumento per il fondo ambientale
 
dt_scint = 10   # ampiezza finestra [s], uguale per entrambe le acquisizioni
 
valori_scint = []   # [0] = orizzontale, [1] = verticale
 
for i_file in range(5, 8):   # 5=orizzontale aria, 6=verticale, 7=orizzontale+tufo
    passo = dt_scint
 
    count, times = lt.read_multiple_variables(f"lettura{i_file}.txt", n_cols=2)
    times = np.array(times)
 
    val   = []
    start = 0.0
    fine  = passo
    idx   = 0
 
    while idx < len(times):
        conteggio = 0
        while idx < len(times) and times[idx] < fine:
            if times[idx] >= start:
                conteggio += 1
            idx += 1
        val.append(conteggio)
        start = fine
        fine += passo
 
    # Porta il vettore a 60 finestre minime aggiungendo zeri se necessario.
    # (Le finestre in coda senza eventi sono comunque tempo morto reale.)
    n_finestre_min = 60
    if len(val) < n_finestre_min:
        val += [0] * (n_finestre_min - len(val))
 
    valori_scint.append(np.array(val))
 
# Stima di λ e del rate per le due configurazioni
rates_scint      = []
sigma_rates_scint = []
 
for val in valori_scint:
    n    = len(val)
    lam  = np.mean(val)
    slam = np.sqrt(lam / n)
    rates_scint.append(lam / dt_scint)
    sigma_rates_scint.append(slam / dt_scint)
 
r_orizz,      sigma_r_orizz      = rates_scint[0], sigma_rates_scint[0]
r_vert,       sigma_r_vert       = rates_scint[1], sigma_rates_scint[1]
r_orizz_tufo, sigma_r_orizz_tufo = rates_scint[2], sigma_rates_scint[2]
 
# ── separazione cosmici / ambientale ─────────────────────────────────────────
r_cosmici       = 2 * (r_orizz - r_vert)
sigma_r_cosmici = 2 * np.sqrt(sigma_r_orizz**2 + sigma_r_vert**2)
 
r_amb_effettivo       = 2 * r_vert - r_orizz          # = R_amb · η  [s⁻¹]
sigma_r_amb_effettivo = np.sqrt(4*sigma_r_vert**2 + sigma_r_orizz**2)
 
r_ambientale_vero       = r_amb_effettivo / ETA_AMBIENTALE
sigma_r_ambientale_vero = sigma_r_amb_effettivo / ETA_AMBIENTALE
 
# ── contributo radioattivo del tufo ──────────────────────────────────────────
# La differenza orizzontale-tufo − orizzontale-aria isola il solo termine R_tufo·η.
r_tufo_effettivo       = r_orizz_tufo - r_orizz
sigma_r_tufo_effettivo = np.sqrt(sigma_r_orizz_tufo**2 + sigma_r_orizz**2)
 
r_tufo_vero       = r_tufo_effettivo / ETA_AMBIENTALE
sigma_r_tufo_vero = sigma_r_tufo_effettivo / ETA_AMBIENTALE
 
print("\n" + "─" * 55)
print("  Scintillatore — separazione rate cosmico / ambientale")
print("─" * 55)
print(f"  R orizzontale (aria)   : {r_orizz:.4f} ± {sigma_r_orizz:.4f} s⁻¹")
print(f"  R verticale   (aria)   : {r_vert:.4f} ± {sigma_r_vert:.4f} s⁻¹")
print(f"  R orizzontale (tufo)   : {r_orizz_tufo:.4f} ± {sigma_r_orizz_tufo:.4f} s⁻¹")
print("─" * 55)
print(f"  R cosmici              : {r_cosmici:.4f} ± {sigma_r_cosmici:.4f} s⁻¹")
print(f"  R ambientale (·η)      : {r_amb_effettivo:.4f} ± {sigma_r_amb_effettivo:.4f} s⁻¹")
print(f"  R ambientale vero      : {r_ambientale_vero:.4f} ± {sigma_r_ambientale_vero:.4f} s⁻¹")
print("─" * 55)
print(f"  R tufo (·η)            : {r_tufo_effettivo:.4f} ± {sigma_r_tufo_effettivo:.4f} s⁻¹")
print(f"  R tufo vero            : {r_tufo_vero:.4f} ± {sigma_r_tufo_vero:.4f} s⁻¹")
print("─" * 55)


# ─────────────────────────────────────────────────────────────────────────────
#  1.7 — Pallinometro: stima MLE di p e test chi²
#
#  Modello: ogni pallina fa N_URTI scelte binarie indipendenti (prob. p di
#  andare a destra), quindi la posizione finale segue una Binomiale(N_URTI, p).
#
#  Stima MLE di p (Formula 15.61):
#    p_hat = (1/N) * Σ f_k * k       dove f_k = m_k / M (frazione nel bin k)
#
#  Incertezza su p_hat dalla curvatura della log-likelihood (Formula 15.63):
#    d²lnL/dp² = − Σ m_k [ k/p² + (N−k)/(1−p)² ]  valutata in p_hat
#    σ²(p_hat) = − 1 / (d²lnL/dp²)
#
#  Test χ² per verificare la compatibilità dei dati con la binomiale teorica:
#    statistica χ² = Σ (osservato − atteso)² / atteso
#    (solo sui bin con atteso ≥ 5; gradi di libertà = n_bin_usati − 1 − 1)
# ─────────────────────────────────────────────────────────────────────────────

def calcola_pallinometro(conteggi, p_teorica=None, n_sim=1_000_000):
    N     = len(conteggi)
    n_tot = conteggi.sum()
    k     = np.arange(N)
    f_k   = conteggi / n_tot

    # ── MLE di p ─────────────────────────────────────────────────────────────
    p_hat = (1 / N_URTI) * np.sum(f_k * k)

    if p_hat <= 0 or p_hat >= 1:
        raise ValueError(f"p_hat = {p_hat:.4f} fuori dal dominio aperto (0, 1)")

    termine = k / p_hat**2 + (N_URTI - k) / (1 - p_hat)**2
    d2_logL = -np.sum(conteggi * termine)
    p_std   = np.sqrt(-1 / d2_logL)

    p_uso = p_teorica if p_teorica is not None else p_hat

    # ── simulazione ──────────────────────────────────────────────────────────
    sim_positions = RNG.binomial(N_URTI, p_uso, size=n_sim)
    sim_freq      = np.bincount(sim_positions, minlength=N) / n_sim

    # ── binomiale teorica ────────────────────────────────────────────────────
    teoria_prob = binom.pmf(k, N_URTI, p_uso)   # <-- definita QUI

    # ── test χ² con accorpamento simmetrico dalle code (atteso ≥ 10) ─────────
    atteso_full = n_tot * teoria_prob
    MIN_ATTESO  = 10

    obs_list = list(conteggi)
    att_list = list(atteso_full)

    # Accorpa dalla coda sinistra (al vicino di destra)
    while len(att_list) > 1 and att_list[0] < MIN_ATTESO:
        val_obs = obs_list.pop(0)
        obs_list[0] += val_obs
        val_att = att_list.pop(0)
        att_list[0] += val_att

    # Accorpa dalla coda destra (al vicino di sinistra)
    while len(att_list) > 1 and att_list[-1] < MIN_ATTESO:
        val_obs = obs_list.pop(-1)
        obs_list[-1] += val_obs
        val_att = att_list.pop(-1)
        att_list[-1] += val_att

    obs_merged = np.array(obs_list)
    att_merged = np.array(att_list)

    k_stimati = 1 if p_teorica is None else 0
    ndof      = len(obs_merged) - 1 - k_stimati
    chi2_val  = np.sum((obs_merged - att_merged)**2 / att_merged)
    p_value = chi2_dist.sf(chi2_val, ndof)

    return dict(conteggi=conteggi, N=N, n_tot=n_tot,
                p_hat=p_hat, p_std=p_std,
                p_uso=p_uso, p_teorica=p_teorica,
                sim_freq=sim_freq, teoria_prob=teoria_prob,
                chi2=chi2_val, ndof=ndof, p_value=p_value,
                obs_merged=obs_merged, att_merged=att_merged)
 
risultati_caso1  = calcola_pallinometro(CONTEGGI_CASO1, p_teorica=0.5)
risultati_caso11 = calcola_pallinometro(CONTEGGI_CASO1, p_teorica=None)
risultati_caso2  = calcola_pallinometro(CONTEGGI_CASO2, p_teorica=None)
 
# Stampa riepilogo pallinometro
for nome, res in [("Caso 1 orizzontale", risultati_caso1),
                  ("Caso 1 orizzontale per binomiale", risultati_caso11),
                  ("Caso 2 inclinato",   risultati_caso2)]:
    print(f"\n{'─'*55}")
    print(f"  {nome}")
    print(f"{'─'*55}")
    print(f"  p_hat  (MLE)        : {res['p_hat']:.5f} ± {res['p_std']:.5f}")
    print(f"  χ² = {res['chi2']:.3f}  ndof = {res['ndof']}  p-value = {res['p_value']:.2e}")
    print(f"\n  Bin accorpati ({len(res['obs_merged'])} bin totali):")
    print(f"  {'#bin':>5}  {'Osservato':>10}  {'Atteso':>10}  {'(O-A)²/A':>10}")
    for i, (o, a) in enumerate(zip(res['obs_merged'], res['att_merged'])):
        print(f"  {i+1:>5}  {o:>10.1f}  {a:>10.3f}  {(o-a)**2/a:>10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
 

# ═════════════════════════════════════════════════════════════════════════════
#  PARTE 2 — GRAFICI
# ═════════════════════════════════════════════════════════════════════════════
 
# ─────────────────────────────────────────────────────────────────────────────
#  2.1 — Istogrammi dei conteggi per finestra (uno per ogni Δt)
#
#  Per ogni misura:
#    • barre blu  = frequenza osservata di ciascun conteggio k
#    • punti rossi = frequenza attesa da Poisson(λ_best) con barre d'errore
#
#  L'incertezza sulla frequenza attesa combina in quadratura:
#    1) varianza binomiale:  n · P(k) · (1 − P(k))
#    2) propagazione di σ(r_best) su λ:
#         σ(n·P) = n · |∂P/∂λ| · σ_λ   con ∂P/∂λ = P(k;λ) · (k/λ − 1)
# ─────────────────────────────────────────────────────────────────────────────

for val, passo in zip(all_vals, dt):
    n           = len(val)
    lam_best    = r_best     * passo
    lam_err     = r_best_err * passo

    k_max  = max(val.max() + 2, int(lam_best * 3 + 1))
    k_vals = np.arange(0, k_max + 1)

    # ── conteggi per bin (sostituisce ax.hist) ────────────────────────────
    bin_counts = np.bincount(val, minlength=k_max + 1)[:k_max + 1]
    
    p_k        = bin_counts / n          # frequenza osservata per ogni bin
    bin_errors = np.sqrt(n * p_k * (1 - p_k))

    # ── previsione teorica con incertezza ─────────────────────────────────
    prob     = poisson.pmf(k_vals, lam_best)
    expected = n * prob
    dp_dlam  = prob * (k_vals / lam_best - 1)
    pred_err = np.abs(n * dp_dlam * lam_err)

    fig, ax = plt.subplots(figsize=(7, 5))

    # barre blu con errore poissoniano
    ax.bar(k_vals, bin_counts, width=0.6,
           color="steelblue", alpha=0.7, label="Dati osservati", zorder=2)
    ax.errorbar(k_vals, bin_counts, yerr=bin_errors,
                fmt="none", color="steelblue", capsize=4,
                elinewidth=1.2, capthick=1.2, zorder=3)

    # punti rossi con errore sulla previsione
    ax.errorbar(k_vals, expected, yerr=pred_err,
                fmt="o-", color="crimson", ms=5, capsize=4,
                label=f"Poisson($\\lambda_{{best}} = {lam_best:.3f}$)", zorder=4)

    ax.set_title(f"Conteggi per finestra — $\\Delta t = {passo}$ s", fontsize=13)
    ax.set_xlabel("Conteggi $X_i$ per finestra", fontsize=11)
    ax.set_ylabel("Numero di finestre", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.savefig(f"Output/jpg/istogramma_dt{passo}s.jpg")
    plt.savefig(f"Output/pdf/istogramma_dt{passo}s.pdf")
    plt.close()     
# ─────────────────────────────────────────────────────────────────────────────
#  2.2 — Probabilità di 0 conteggi in funzione di Δt
#
#  Punti blu con barre d'errore = stime sperimentali f0 ± σ(f0)
#  Curva rossa = previsione teorica e^{−r_best · Δt}
# ─────────────────────────────────────────────────────────────────────────────
 
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.errorbar(dt, f0_obs, yerr=f0_err,
             fmt="o", color="steelblue", capsize=5, capthick=1.5,
             elinewidth=1.5, ms=7, label="Osservato")
ax1.plot(t_fine, p0_teoria, color="crimson", lw=2,
         label=f"$e^{{-r_{{best}}\\,\\Delta t}}$  ($r_{{best}} = {r_best:.4f}$ s⁻¹)")
ax1.set_xlabel("$\\Delta t$ [s]", fontsize=12)
ax1.set_ylabel("Probabilità stimata $\\hat{P}(X=0)$", fontsize=12)
ax1.set_title("Probabilità di 0 conteggi in funzione di $\\Delta t$", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.4)
fig1.tight_layout()
plt.savefig("Output/jpg/prob_zero.jpg")
plt.savefig("Output/pdf/prob_zero.pdf")
plt.close()
 
# ─────────────────────────────────────────────────────────────────────────────
#  2.3 — Probabilità di 0 o 1 conteggio in funzione di Δt
#
#  Punti blu = stime sperimentali f01 ± σ(f01)
#  Curva rossa = (1 + r_best · Δt) · e^{−r_best · Δt}
# ─────────────────────────────────────────────────────────────────────────────
 
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.errorbar(dt, f01_obs, yerr=f01_err,
             fmt="o", color="steelblue", capsize=5, capthick=1.5,
             elinewidth=1.5, ms=7, label="Osservato")
ax2.plot(t_fine, p01_teoria, color="crimson", lw=2,
         label=f"$(1 + r_{{best}}\\,\\Delta t)\\,e^{{-r_{{best}}\\,\\Delta t}}$"
               f"  ($r_{{best}} = {r_best:.4f}$ s⁻¹)")
ax2.set_xlabel("$\\Delta t$ [s]", fontsize=12)
ax2.set_ylabel("Probabilità stimata $\\hat{P}(X \\leq 1)$", fontsize=12)
ax2.set_title("Probabilità di 0 o 1 conteggio in funzione di $\\Delta t$", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.4)
fig2.tight_layout()
plt.savefig("Output/jpg/prob_zero_uno.jpg")
plt.savefig("Output/pdf/prob_zero_uno.pdf")
plt.close()
 
# ─────────────────────────────────────────────────────────────────────────────
#  2.4 — Posterior pdf di r (tre casi: Δt=1s, Δt=5s, tutte)
#
#  Per ogni caso:
#    • curva blu      = densità della Gamma posterior
#    • area ombreggiata = intervallo 68.3% intorno alla media
#    • linea rossa    = media posteriore
#    • linea arancione = mediana posteriore
#    • linea verde    = r_best (media pesata classica, per confronto)
# ─────────────────────────────────────────────────────────────────────────────
 
r_ax = np.linspace(1e-6, 3 * r_best, 2000)
 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (nome, res) in zip(axes, casi_posterior.items()):
    dist     = res["dist"]
    media    = res["media"]
    mediana  = res["mediana"]
    lo, hi   = res["intervallo_683"]
 
    ax.plot(r_ax, dist.pdf(r_ax), color="steelblue", lw=2, label="Posterior")
    r_fill = np.linspace(lo, hi, 500)
    ax.fill_between(r_fill, dist.pdf(r_fill),
                    color="steelblue", alpha=0.25, label="68.3% intorno a $\\mu$")
    ax.axvline(media,   color="crimson",    lw=1.5, ls="-",  label=f"Media = {media:.4f}")
    ax.axvline(mediana, color="darkorange", lw=1.5, ls="--", label=f"Mediana = {mediana:.4f}")
    ax.axvline(r_best,  color="green",      lw=1.5, ls=":",  label=f"$r_{{best}}$ = {r_best:.4f}")
    ax.set_title(f"Posterior — {nome}", fontsize=12)
    ax.set_xlabel("$r$ [s⁻¹]", fontsize=11)
    ax.set_ylabel("$p(r \\mid \\mathrm{dati})$", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
 
fig.suptitle("Posterior pdf di $r$ — inferenza Poissoniana con prior uniforme",
             fontsize=13)
fig.tight_layout()
plt.savefig("Output/jpg/posterior_r.jpg")
plt.savefig("Output/pdf/posterior_r.pdf")
plt.close()
 
# ─────────────────────────────────────────────────────────────────────────────
#  2.5 — Tempi di attesa: confronto con Erlang(k, r_best)
#
#  Per ogni ordine k:
#    • istogramma normalizzato a densità = distribuzione osservata dei τ
#    • curva rossa = pdf teorica Erlang(k, r_best)
#    • linee tratteggiate = media teorica e media osservata
# ─────────────────────────────────────────────────────────────────────────────
 
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, k in zip(axes.flat, k_list):
    tau   = tau_per_k[k]
    stats = statistiche_tau[k]
 
    ax.hist(tau, bins="auto", density=True,
            color="steelblue", alpha=0.7, label=f"Dati ($k={k}$)")
 
    t_fine_k = np.linspace(0, tau.max() * 1.2, 1000)
    ax.plot(t_fine_k, erlang.pdf(t_fine_k, a=k, scale=1 / r_best),
            color="crimson", lw=2,
            label=("Exp($r_{best}$)" if k == 1
                   else f"Erlang($k={k}$, $r_{{best}}$)"))
 
    ax.axvline(stats["media_teoria"], color="crimson",   ls="--", lw=1.2,
               label=f"$\\mu_{{teo}} = {stats['media_teoria']:.2f}$ s")
    ax.axvline(stats["media_obs"],    color="steelblue", ls="--", lw=1.2,
               label=f"$\\bar{{\\tau}} = {stats['media_obs']:.2f}$ s")
 
    ax.set_title(f"Tempi di attesa — $k = {k}$", fontsize=12)
    ax.set_xlabel("Tempo di attesa $\\tau$ [s]", fontsize=11)
    ax.set_ylabel("Densità", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
 
fig.suptitle("Tempi di attesa — confronto con Erlang($k$, $r_{best}$)", fontsize=13)
fig.tight_layout()
plt.savefig("Output/jpg/erlang_waiting_times.jpg")
plt.savefig("Output/pdf/erlang_waiting_times.pdf")
plt.close()
 
# ─────────────────────────────────────────────────────────────────────────────
#  2.6 — Pallinometro: distribuzioni e likelihood di p (due casi)
#
#  Pannello sinistro (più largo):
#    • barre colorate = dati reali normalizzati a probabilità
#    • barre arancioni = frequenze dalla simulazione (10⁶ palline)
#    • punti rossi    = binomiale teorica Binom(N_URTI, p_uso)
#    • testo in alto a destra = risultato del test χ²
#
#  Pannello destro — approssimazione gaussiana della likelihood intorno a p_hat:
#    Per M grande la log-likelihood è parabolica intorno al massimo, quindi
#    la likelihood normalizzata si approssima con N(p_hat, σ(p_hat)).
#    • curva blu      = gaussiana N(p_hat, σ)
#    • area ombreggiata = intervallo ±1σ (68.3%)
#    • linea rossa    = p_hat MLE
#    • linea verde    = p teorica (solo caso 1, dove p_teo = 0.5 è nota)
# ─────────────────────────────────────────────────────────────────────────────
 
from scipy.stats import norm as norm_dist
 
casi_pall = [
    ("Caso 1 orizzontale", risultati_caso1, "steelblue"),
    ("Caso 1 orizzontale con p sperimentale", risultati_caso11, "teal"),
    ("Caso 2 inclinato",   risultati_caso2, "darkorchid"),
]
 
for nome_caso, res, colore in casi_pall:
    N           = res["N"]
    n_tot       = res["n_tot"]
    conteggi    = res["conteggi"]
    k           = np.arange(N)
    p_hat       = res["p_hat"]
    p_std       = res["p_std"]
    p_uso       = res["p_uso"]
    p_teorica   = res["p_teorica"]
    sim_freq    = res["sim_freq"]
    teoria_prob = res["teoria_prob"]
    chi2_val    = res["chi2"]
    ndof        = res["ndof"]
    p_value     = res["p_value"]
 
    fig, (ax_main, ax_post) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [2, 1]}
    )
 
    # pannello sinistro: distribuzioni
    larghezza = 0.3
    ax_main.bar(k - larghezza, conteggi / n_tot, width=larghezza,
                color=colore, alpha=0.8, label=f"Dati reali (n={n_tot})")
    ax_main.bar(k, sim_freq, width=larghezza,
                color="darkorange", alpha=0.7, label="Simulazione ($10^6$ palline)")
    ax_main.plot(k, teoria_prob, "o-", color="crimson", ms=6, lw=1.5,
                 label=f"Binomiale($p={p_uso:.4f}$, $n={N_URTI}$)")
    ax_main.text(0.98, 0.97,
                 f"$\\chi^2 = {chi2_val:.2f}$\nndof $= {ndof}$\n$p$-value $= {p_value:.5f}$",
                 transform=ax_main.transAxes, fontsize=9, va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    ax_main.set_title(f"{nome_caso} — confronto distribuzioni", fontsize=12)
    ax_main.set_xlabel("Bin (posizione finale)", fontsize=11)
    ax_main.set_ylabel("Probabilità", fontsize=11)
    ax_main.set_xticks(k)
    ax_main.legend(fontsize=9)
    ax_main.grid(True, linestyle="--", alpha=0.4)
 
    # pannello destro: approssimazione gaussiana della likelihood
    gauss    = norm_dist(loc=p_hat, scale=p_std)
    p_ax     = np.linspace(max(0, p_hat - 5*p_std), min(1, p_hat + 5*p_std), 1000)
    p_fill   = np.linspace(p_hat - p_std, p_hat + p_std, 500)
 
    ax_post.plot(p_ax, gauss.pdf(p_ax), color="steelblue", lw=2,
                 label="Likelihood (approx. gaussiana)")
    ax_post.fill_between(p_fill, gauss.pdf(p_fill),
                         color="steelblue", alpha=0.25,
                         label=f"$\\pm 1\\sigma$: [{p_hat-p_std:.4f}, {p_hat+p_std:.4f}]")
    ax_post.axvline(p_hat, color="crimson", lw=1.5, ls="-",
                    label=f"$\\hat{{p}}_{{MLE}} = {p_hat:.4f} \\pm {p_std:.4f}$")
    if p_teorica is not None:
        ax_post.axvline(p_teorica, color="green", lw=1.5, ls="--",
                        label=f"$p_{{teo}} = {p_teorica}$")
    ax_post.set_title(f"Likelihood di $p$ — {nome_caso}", fontsize=12)
    ax_post.set_xlabel("$p$", fontsize=11)
    ax_post.set_ylabel("Densità (approx.)", fontsize=11)
    ax_post.legend(fontsize=8)
    ax_post.grid(True, linestyle="--", alpha=0.4)
 
    fig.tight_layout()
    tag = nome_caso.lower().replace(" ", "_")
    plt.savefig(f"Output/jpg/pallinometro_{tag}.jpg")
    plt.savefig(f"Output/pdf/pallinometro_{tag}.pdf")
    plt.show()
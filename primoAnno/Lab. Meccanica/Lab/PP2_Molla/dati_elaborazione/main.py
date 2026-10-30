import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import lab_tools as lt
plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# COSTANTI DI RISOLUZIONE STRUMENTI
# ==============================================================================
RES_MASSA = 0.0001      # Bilancia: 0.1 g = 0.0001 kg
RES_TEMPO = 0.01        # Cronometro: 0.01 s
RES_LUNGHEZZA = 0.001   # Metro: 1 mm = 0.001 m
offset_tempo = 0.1

# ============================================================================
# PARTE 1: ANALISI PRELIMINARE (METODO A: 5 osc, METODO B: 50 osc)
# ============================================================================

# X_eq5 = 44 / 1000       # Posizione equilibrio per m5 (44 mm = 0.044 m)
# X_eq10 = 126 / 1000     # Posizione equilibrio per m10 (126 mm = 0.126 m)

file_input = "oscillazioni.txt"

# Liste temporanee per raccogliere i tempi dai 4 gruppi
g5A, g5B, g10A, g10B = [], [], [], []

# Leggiamo il file riga per riga
with open(file_input, 'r') as f:
    for line in f:
        valori = line.split()  # Divide la riga per spazi
        
        if len(valori) == 4:
            g5A.append(float(valori[0]))
            g5B.append(float(valori[1]))
            g10A.append(float(valori[2]))
            g10B.append(float(valori[3]))
        elif len(valori) == 2:
            g5A.append(float(valori[0]))
            g10A.append(float(valori[1]))

# Convertiamo liste in array NumPy (Tempi totali per l'analisi successiva)
gruppo5A = np.array(g5A)   + offset_tempo
gruppo10A = np.array(g10A) + offset_tempo
gruppo5B = np.array(g5B)   + offset_tempo
gruppo10B = np.array(g10B) + offset_tempo

# Calcola statistiche campionarie sui tempi totali (serve per i calcoli successivi)
risultati5A = lt.sample_statistic(gruppo5A)
risultati5B = lt.sample_statistic(gruppo5B)
risultati10A = lt.sample_statistic(gruppo10A)
risultati10B = lt.sample_statistic(gruppo10B)

# --- CREAZIONE ARRAY DEI PERIODI SOLO PER IL PLOTTING ---
periodo5A_plot  = (gruppo5A) / 5
periodo10A_plot = (gruppo10A) / 5
periodo5B_plot  = (gruppo5B) / 50
periodo10B_plot = (gruppo10B) / 50

# Statistiche per i plot
stat_plot_5A = lt.sample_statistic(periodo5A_plot)
stat_plot_5B = lt.sample_statistic(periodo5B_plot)
stat_plot_10A = lt.sample_statistic(periodo10A_plot)
stat_plot_10B = lt.sample_statistic(periodo10B_plot)

# Creazione figura con 4 istogrammi + gaussiane sovrapposte
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

dati_tutti_plot = [periodo5A_plot, periodo5B_plot, periodo10A_plot, periodo10B_plot]
risultati_tutti_plot = [stat_plot_5A, stat_plot_5B, stat_plot_10A, stat_plot_10B]
labels = ["Gruppo 5A (Periodo)", "Gruppo 5B (Periodo)", "Gruppo 10A (Periodo)", "Gruppo 10B (Periodo)"]
colori_hist = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
colori_gauss = ['#1F4E78', '#C65911', '#404040', '#997300']

for i in range(4):
    mean, var, std, std_mean, N = risultati_tutti_plot[i]
    
    # Calcolo bin con formula di Freedman-Diaconis
    q75, q25 = np.percentile(dati_tutti_plot[i], [75, 25])
    iqr = q75 - q25
    if iqr > 0:
        bin_width = 2 * iqr * N**(-1/3)
        bins = int(np.ceil((dati_tutti_plot[i].max() - dati_tutti_plot[i].min()) / bin_width))
    else:
        bins = int(np.ceil(np.log2(N) + 1))
    
    # Istogramma normalizzato
    axs[i].hist(dati_tutti_plot[i], bins=bins, edgecolor='black', color=colori_hist[i], 
                alpha=0.75, density=True, label='Dati osservati', linewidth=1.2)
    
    # Gaussiana teorica sovrapposta
    x_min, x_max = dati_tutti_plot[i].min(), dati_tutti_plot[i].max()
    x = np.linspace(x_min - 2.5*std, x_max + 2.5*std, 500)
    gaussiana = norm.pdf(x, mean, std)
    axs[i].plot(x, gaussiana, color=colori_gauss[i], linewidth=3, 
                label=f'Gaussiana: μ={mean:.3f}, σ={std:.3f}')
    
    # Linee di riferimento
    axs[i].axvline(mean, color=colori_gauss[i], linestyle='--', linewidth=2, alpha=0.7, label=f'Media')
    axs[i].axvline(mean - std, color=colori_gauss[i], linestyle=':', linewidth=1.5, alpha=0.5)
    axs[i].axvline(mean + std, color=colori_gauss[i], linestyle=':', linewidth=1.5, alpha=0.5, 
                   label=f'μ ± σ')
    
    # Personalizzazione assi
    titolo = f"{labels[i]}\nn = {N} | Media = {mean:.4f} s | σ = {std:.4f} s | σ_media = {std_mean:.4f} s | bins = {bins}"
    axs[i].set_title(titolo, fontsize=11, fontweight='bold', pad=12)
    axs[i].set_xlabel("Periodo dell'oscillazione $T$ [s]", fontsize=10, fontweight='bold')
    axs[i].set_ylabel("Densità di probabilità", fontsize=10, fontweight='bold')
    axs[i].grid(axis='y', linestyle='-', alpha=0.3, linewidth=0.7)
    axs[i].set_axisbelow(True)
    axs[i].legend(loc='upper right', fontsize=9, framealpha=0.95)
    axs[i].tick_params(labelsize=9)

fig.suptitle('Analisi Statistica Oscillazioni (Periodo) - Istogrammi con Gaussiane Sovrapposte', 
             fontsize=14, fontweight='bold', y=0.995)

plt.subplots_adjust(
    top=0.88,     # Lascia abbondante spazio in alto per il titolo principale (suptitle)
    bottom=0.1,   # Margine inferiore della figura
    left=0.08,    # Margine sinistro
    right=0.95,   # Margine destro
    hspace=0.45,  # SPAZIO VERTICALE tra la riga sopra e quella sotto (aumentalo a 0.5 o 0.6 se serve)
    wspace=0.2   # SPAZIO ORIZZONTALE tra la colonna destra e sinistra
)
plt.savefig("oscillazioni_periodi.png", dpi=300, bbox_inches='tight')
plt.show()

# Calcola incertezze di risoluzione
sigma_b_tempo     = lt.resolution_uncertainty(RES_TEMPO)
sigma_b_massa     = lt.resolution_uncertainty(RES_MASSA)
sigma_b_lunghezza = lt.resolution_uncertainty(RES_LUNGHEZZA)

X_eq5 = lt.read_single_variable("allungamento_5.txt") / 1000
X_eq10 = lt.read_single_variable("allungamento_10.txt")  / 1000

risultati_x_eq_5 = lt.sample_statistic(X_eq5)
risultati_x_eq_10 = lt.sample_statistic(X_eq10)

lt.print_sample_statistics(risultati_x_eq_5)
lt.print_sample_statistics(risultati_x_eq_10)

X_eq5 = risultati_x_eq_5[0]
X_eq10 =  risultati_x_eq_10[0]

risultati_finali_allungamento_5 =lt.get_total_uncertainty(X_eq5, risultati_x_eq_5[3], sigma_b_lunghezza)
risultati_finali_allungamento_10 =lt.get_total_uncertainty(X_eq10, risultati_x_eq_10[3], sigma_b_lunghezza)

sigma_totale_lunghezza = risultati_finali_allungamento_5['sigma_total']

# Leggi le masse per m5 e m10 da file
masse_10, masse_5, _ = lt.read_xy_data("masse.txt")

# Converti da grammi a chilogrammi
masse_10 /= 1000
masse_5 /= 1000

# Calcola statistiche delle masse
risultati_masse_10 = lt.sample_statistic(masse_10)
risultati_masse_5 = lt.sample_statistic(masse_5)

# Estrai medie
massa_10 = risultati_masse_10[0]
massa_5  = risultati_masse_5[0]

# Estrai tempi medi
tempo_10_A = risultati10A[0]
tempo_5_A  = risultati5A[0]
tempo_10_B = risultati10B[0]
tempo_5_B  = risultati5B[0]

# Calcola periodi
periodo_10_A = tempo_10_A / 5
periodo_5_A  = tempo_5_A / 5
periodo_10_B = tempo_10_B / 50
periodo_5_B  = tempo_5_B / 50

# Combina incertezze
risultati_tempo_10_A = lt.get_total_uncertainty(tempo_10_A, risultati10A[3], sigma_b_tempo)
risultati_tempo_5_A  = lt.get_total_uncertainty(tempo_5_A, risultati5A[3], sigma_b_tempo)
risultati_tempo_10_B = lt.get_total_uncertainty(tempo_10_B, risultati10B[3], sigma_b_tempo)
risultati_tempo_5_B  = lt.get_total_uncertainty(tempo_5_B, risultati5B[3], sigma_b_tempo)


lt.print_total_uncertainty(risultati_tempo_10_A['mean'], risultati_tempo_10_A['sigma_a'], risultati_tempo_10_A['sigma_b'], risultati_tempo_10_A['sigma_total'])
lt.print_total_uncertainty(risultati_tempo_5_A['mean'], risultati_tempo_5_A['sigma_a'], risultati_tempo_5_A['sigma_b'], risultati_tempo_5_A['sigma_total'])
lt.print_total_uncertainty(risultati_tempo_10_B['mean'], risultati_tempo_10_B['sigma_a'], risultati_tempo_10_B['sigma_b'], risultati_tempo_10_B['sigma_total'])
lt.print_total_uncertainty(risultati_tempo_5_B['mean'], risultati_tempo_5_B['sigma_a'], risultati_tempo_5_B['sigma_b'], risultati_tempo_5_B['sigma_total'])

risultati_massa_5  = lt.get_total_uncertainty(massa_5, risultati_masse_5[3], sigma_b_massa)
risultati_massa_10 = lt.get_total_uncertainty(massa_10, risultati_masse_10[3], sigma_b_massa)

# Estrai incertezze totali
sigma_massa_5 = risultati_massa_5['sigma_total']
sigma_massa_10 = risultati_massa_10['sigma_total']
sigma_tempo_10_A = risultati_tempo_10_A['sigma_total']
sigma_tempo_5_A = risultati_tempo_5_A['sigma_total']
sigma_tempo_10_B = risultati_tempo_10_B['sigma_total']
sigma_tempo_5_B = risultati_tempo_5_B['sigma_total']

# Propaga incertezze su periodi
sigma_periodo_10_A = sigma_tempo_10_A / 5
sigma_periodo_5_A  = sigma_tempo_5_A  / 5
sigma_periodo_10_B = sigma_tempo_10_B / 50
sigma_periodo_5_B  = sigma_tempo_5_B  / 50

print("PERIODO 10_A")
lt.print_total_uncertainty(periodo_10_A, 0, 0, sigma_periodo_10_A)

print("PERIODO 5_A")
lt.print_total_uncertainty(periodo_5_A, 0, 0, sigma_periodo_5_A)

print("PERIODO 10_B")
lt.print_total_uncertainty(periodo_10_B, 0, 0, sigma_periodo_10_B)

print("PERIODO 5_B")
lt.print_total_uncertainty(periodo_5_B, 0, 0, sigma_periodo_5_B)

print("MASSE 5")
lt.print_total_uncertainty(risultati_massa_5['mean'], risultati_massa_5['sigma_a'],
                          risultati_massa_5['sigma_b'], risultati_massa_5['sigma_total'])

print("MASSE 10")
lt.print_total_uncertainty(risultati_massa_10['mean'], risultati_massa_10['sigma_a'],
                          risultati_massa_10['sigma_b'], risultati_massa_10['sigma_total'])

# ==============================================================================
# CALCOLO DI k E g - METODI A e B
# ==============================================================================

k_A = 4 * (np.pi)**2 * (massa_10 - massa_5) / (periodo_10_A**2 - periodo_5_A**2)
k_B = 4 * (np.pi)**2 * (massa_10 - massa_5) / (periodo_10_B**2 - periodo_5_B**2)

chi_A = (periodo_10_A**2 - periodo_5_A**2)
chi_B = (periodo_10_B**2 - periodo_5_B**2)
dm = massa_10 - massa_5

sigma_k_A = 4*np.pi**2 / chi_A * np.sqrt(
    sigma_massa_10**2 + sigma_massa_5**2 + 
    dm**2 / chi_A**2 * (sigma_periodo_10_A**2 + 
            sigma_periodo_5_A**2)
)

sigma_k_B = 4*np.pi**2 / chi_B * np.sqrt(
    sigma_massa_10**2 + sigma_massa_5**2 + 
    dm**2 / chi_B**2 * (sigma_periodo_10_B**2 + 
            sigma_periodo_5_B**2)
)


dx = X_eq10 - X_eq5

g_sperimentale_A = 4 * (np.pi)**2 * dx / (periodo_10_A**2 - periodo_5_A**2)
g_sperimentale_B = 4 * (np.pi)**2 * dx / (periodo_10_B**2 - periodo_5_B**2)

sigma_g_A = 4*np.pi**2 / chi_A * np.sqrt(
    (sigma_totale_lunghezza**2 + sigma_totale_lunghezza**2) + 
    dx**2 / chi_A**2 * (sigma_periodo_10_A**2 + 
            sigma_periodo_5_A**2)
)

sigma_g_B = 4*np.pi**2 / chi_B * np.sqrt(
    (sigma_totale_lunghezza**2 + sigma_totale_lunghezza**2) + 
    dx**2 / chi_B**2 * (sigma_periodo_10_B**2 + 
            sigma_periodo_5_B**2)
)

print("\n" + "="*80)
print("RISULTATI METODO B (50 oscillazioni)")
print("="*80)
print(f"Costante elastica:           k = {k_B} ± {sigma_k_B} N/m")
print(f"Accelerazione di gravità:    g = {g_sperimentale_B} ± {sigma_g_B} m/s²")

print("\n" + "="*80)
print("RISULTATI METODO A (5 oscillazioni)")
print("="*80)
print(f"Costante elastica:           k = {k_A} ± {sigma_k_A} N/m")
print(f"Accelerazione di gravità:    g = {g_sperimentale_A} ± {sigma_g_A} m/s²")

# Grafico di confronto
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

g_min = min(g_sperimentale_A - 4*sigma_g_A, g_sperimentale_B - 4*sigma_g_B)
g_max = max(g_sperimentale_A + 4*sigma_g_A, g_sperimentale_B + 4*sigma_g_B)
x_g = np.linspace(g_min, g_max, 1000)

y_g_A = norm.pdf(x_g, g_sperimentale_A, sigma_g_A)
y_g_B = norm.pdf(x_g, g_sperimentale_B, sigma_g_B)

ax1.plot(x_g, y_g_A, label=f'Metodo A (5 osc.): $g={g_sperimentale_A:.2f} \\pm {sigma_g_A:.2f}$', 
         color='#1F4E78', linewidth=2.5)
ax1.fill_between(x_g, y_g_A, alpha=0.2, color='#1F4E78')
ax1.plot(x_g, y_g_B, label=f'Metodo B (50 osc.): $g={g_sperimentale_B:.2f} \\pm {sigma_g_B:.2f}$', 
         color='#C65911', linewidth=2.5)
ax1.fill_between(x_g, y_g_B, alpha=0.2, color='#C65911')

ax1.set_title('Confronto Accelerazione di Gravità $g$', fontsize=13, fontweight='bold')
ax1.set_xlabel('$g$ $[m/s^2]$', fontsize=11)
ax1.set_ylabel('Densità di probabilità', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right', fontsize=9)

k_min = min(k_A - 4*sigma_k_A, k_B - 4*sigma_k_B)
k_max = max(k_A + 4*sigma_k_A, k_B + 4*sigma_k_B)
x_k = np.linspace(k_min, k_max, 1000)

y_k_A = norm.pdf(x_k, k_A, sigma_k_A)
y_k_B = norm.pdf(x_k, k_B, sigma_k_B)

ax2.plot(x_k, y_k_A, label=f'Metodo A (5 osc.): $k={k_A:.2f} \\pm {sigma_k_A:.2f}$', 
         color='#1F4E78', linewidth=2.5)
ax2.fill_between(x_k, y_k_A, alpha=0.2, color='#1F4E78')
ax2.plot(x_k, y_k_B, label=f'Metodo B (50 osc.): $k={k_B:.2f} \\pm {sigma_k_B:.2f}$', 
         color='#C65911', linewidth=2.5)
ax2.fill_between(x_k, y_k_B, alpha=0.2, color='#C65911')

ax2.set_title('Confronto Costante Elastica $k$', fontsize=13, fontweight='bold')
ax2.set_xlabel('$k$ $[N/m]$', fontsize=11)
ax2.set_ylabel('Densità di probabilità', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', fontsize=9)

fig.suptitle('Analisi Comparativa tra Metodo A (5 oscillazioni) e Metodo B (50 oscillazioni)', 
             fontsize=15, fontweight='bold')

plt.savefig("confronto_metodi_g_k.png", dpi=300)
plt.show()


# ============================================================================
# PARTE 2: ANALISI DATI INTEGRALI CON FIT LINEARE MULTIPLO
# ============================================================================

print("\n" + "="*80)
print("FASE 2: ANALISI DATI INTEGRALI CON FIT LINEARE MULTIPLO")
print("="*80)

print("\n[A] CARICAMENTO DATI DA FILE...")

masse_matrix = np.loadtxt('masseIntegrali.txt', unpack=True) / 1000
tempi_matrix = np.loadtxt('tempiIntegrali.txt', unpack=True) + offset_tempo

tempi_matrix[0] -= offset_tempo
tempi_matrix[1] -= offset_tempo

N_OSC = 10

print(f"   Dati caricati: {masse_matrix.shape[0]} misure per {masse_matrix.shape[1]} posizioni")
print(f"   Oscillazioni per misurazione: {N_OSC}")

print("\n[B] DEFINIZIONE POSIZIONI DI EQUILIBRIO...")

xEq1  = 0.0
xEq2  = 0.0
xEq3  = 11.0
xEq4  = 17.5
xEq5  = 43.0
xEq6  = 61.0
xEq7  = 76.0
xEq8  = 93.0
xEq9  = 110.0
xEq10 = 126.0

x_eq_tutti = np.array([xEq1, xEq2, xEq3, xEq4, xEq5, xEq6, xEq7, xEq8, xEq9, xEq10]) / 1000

sigma_tot_x_eq = np.sqrt(sigma_b_lunghezza**2 + risultati_x_eq_5[2]**2)

sigma_x_eq_tutti = np.full_like(x_eq_tutti, sigma_tot_x_eq)

print(f"   Posizioni di equilibrio (in m): {x_eq_tutti}")
print(f"   Incertezza su ogni posizione: {sigma_x_eq_tutti} m")

print("\n[C] ELABORAZIONE STATISTICA (calcolo medie, varianze, incertezze)...")

masse_medie_lista = []
sigma_masse_lista = []
periodi_quadri_lista = []
sigma_periodi_quadri_lista = []

print("\n--- VALORI MEDI E INCERTEZZE (per singolo set/posizione) ---")

for i in range(10):
    stat_massa = lt.sample_statistic(masse_matrix[i])
    massa_media = stat_massa[0]
    sigma_a_massa = stat_massa[3]
    sigma_tot_massa = np.sqrt(sigma_a_massa**2 + sigma_b_massa**2)
    
    stat_tempo = lt.sample_statistic(tempi_matrix[i])
    tempo_totale = stat_tempo[0]
    sigma_a_tempo = stat_tempo[3]
    sigma_tot_tempo = np.sqrt(sigma_a_tempo**2 + sigma_b_tempo**2)
    
    periodo_singolo = (tempo_totale) / N_OSC
    periodo_quadro = periodo_singolo**2
    sigma_periodo_quadro = (2 * periodo_singolo * sigma_tot_tempo) / (N_OSC)
    
    print(f"Set {i+1:2d} | Massa: {massa_media:.5f} ± {sigma_tot_massa:.1e} kg | "
          f"T²: {periodo_quadro:.5f} ± {sigma_periodo_quadro:.1e} s²")
    
    masse_medie_lista.append(massa_media)
    sigma_masse_lista.append(sigma_tot_massa)
    periodi_quadri_lista.append(periodo_quadro)
    sigma_periodi_quadri_lista.append(sigma_periodo_quadro)

print("\n[D] CONVERSIONE IN ARRAY NUMPY E PULIZIA DATI...")

masse_pulite = np.array(masse_medie_lista)
sigma_masse_pulite = np.array(sigma_masse_lista)
periodi_quadri_puliti = np.array(periodi_quadri_lista)
sigma_periodi_quadri_puliti = np.array(sigma_periodi_quadri_lista)

x_eq_tutti[3] += 0.00706

punti_da_rimuovere = 3
print(f"\n   Rimozione dei primi {punti_da_rimuovere} punti")

masse_pulite = masse_pulite[punti_da_rimuovere:]
sigma_masse_pulite = sigma_masse_pulite[punti_da_rimuovere:]
periodi_quadri_puliti = periodi_quadri_puliti[punti_da_rimuovere:]
sigma_periodi_quadri_puliti = sigma_periodi_quadri_puliti[punti_da_rimuovere:]
x_eq_puliti = x_eq_tutti[punti_da_rimuovere:]
sigma_x_eq_puliti = sigma_x_eq_tutti[punti_da_rimuovere:]


print(f"   Punti rimanenti per i fit: {len(masse_pulite)} su 10 originali")

# --------------------------------------------------------------------------
# SEZIONE E: FIT 1 - PERIODO AL QUADRATO vs MASSA
# --------------------------------------------------------------------------

print("\n" + "-"*80)
print("FIT 1: T² vs Massa")
print("-"*80)

risultati_fit_T2_m = lt.linear_fit_complete(masse_pulite, periodi_quadri_puliti, sigma_periodi_quadri_puliti)
lt.print_fit_results(risultati_fit_T2_m)

(pendenza_T2_m, intercetta_T2_m, sigma_pendenza_T2_m, sigma_intercetta_T2_m, 
 covarianza_T2_m, correlazione_T2_m, chi2_T2_m, chi2_ridotto_T2_m, 
 N_dati_T2_m, e_pesato_T2_m) = risultati_fit_T2_m

plt.figure(figsize=(10, 6))
lt.plot_generic_fit(
    masse_pulite, periodi_quadri_puliti, sigma_periodi_quadri_puliti,
    pendenza_T2_m, intercetta_T2_m, sigma_pendenza_T2_m, sigma_intercetta_T2_m, covarianza_T2_m,
    title=f"Fit 1: T² vs Massa ({N_OSC} oscillazioni per misura)",
    xlabel="Massa $m$ [kg]", ylabel="Periodo al quadrato $T^2$ $[s^2]$",
    plot=True, save=False, type="Lineare"
)

plt.tight_layout()
plt.savefig("fit_lineare_T2_vs_m.png", dpi=300)
plt.show()

errore_X_proiettato_Y = np.mean(sigma_masse_pulite) * pendenza_T2_m
errore_Y_medio = np.mean(sigma_periodi_quadri_puliti)
rapporto_errori = errore_X_proiettato_Y / errore_Y_medio

print(f"\nControllo Validità Fit (errori su X proiettati su Y):")
print(f"   Errore X proiettato su Y: {errore_X_proiettato_Y:.2e}")
print(f"   Errore Y medio: {errore_Y_medio:.2e}")
print(f"   Rapporto: {rapporto_errori:.2%}")
if errore_X_proiettato_Y < 0.1 * errore_Y_medio:
    print(f"   ✅ Fit è VALIDO (rapporto < 10%)")
else: 
    print(f"   ⚠️  Fit potrebbe avere errori sistematici")
    
k_fit1 = 4 * np.pi**2 / pendenza_T2_m
sigma_k_fit_1 = k_fit1 / pendenza_T2_m * sigma_pendenza_T2_m

print(f"\nCostante elastica ricavata da Fit 1: k = 4π² / {pendenza_T2_m:.6f} = {k_fit1:.2f} N/m")
print(f"\nIncertezza costante elastica: {sigma_k_fit_1}")

m_0_fit1 = intercetta_T2_m * k_fit1 / (4*np.pi**2)
sigma_m_0_fit1 = 1 / (4 * np.pi**2) * np.sqrt(k_fit1**2 * sigma_intercetta_T2_m**2 + intercetta_T2_m**2 * sigma_k_fit_1**2)
    
print(f"M_0 = c * k / (4π²) = {intercetta_T2_m:.5f} * {k_fit1:.2f} / (4π²) = {m_0_fit1:.5f} kg")
print(f"\nIncertezza m_0: {sigma_m_0_fit1}")

    

# --------------------------------------------------------------------------
# SEZIONE F: FIT 2 - POSIZIONE DI EQUILIBRIO vs MASSA
# --------------------------------------------------------------------------

print("\n" + "-"*80)
print("FIT 2: x_eq vs Massa")
print(f"Incertezza x: {np.mean(sigma_x_eq_puliti)} e m: {np.mean(sigma_masse_pulite)}")
print("-"*80)

risultati_fit_xeq_m = lt.linear_fit_complete(masse_pulite, x_eq_puliti, sigma_x_eq_puliti)
lt.print_fit_results(risultati_fit_xeq_m)

(pendenza_xeq_m, intercetta_xeq_m, sigma_pendenza_xeq_m, sigma_intercetta_xeq_m, 
 covarianza_xeq_m, correlazione_xeq_m, chi2_xeq_m, chi2_ridotto_xeq_m, 
 N_dati_xeq_m, e_pesato_xeq_m) = risultati_fit_xeq_m

plt.figure(figsize=(10, 6))
lt.plot_generic_fit(
    masse_pulite, x_eq_puliti, sigma_x_eq_puliti, 
    pendenza_xeq_m, intercetta_xeq_m, sigma_pendenza_xeq_m, sigma_intercetta_xeq_m, covarianza_xeq_m,
    title="Fit 2: Posizione di equilibrio vs Massa",
    xlabel="Massa $m$ [kg]",
    ylabel="Posizione di equilibrio $x_{eq}$ [m]",
    plot=True, save=False, type="Lineare"
)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("fit_lineare_xeq_vs_m.png", dpi=300)
plt.show()

g_calc = 4 * np.pi**2 * pendenza_xeq_m / pendenza_T2_m
sigma_g_calc = 4 * np.pi**2 / pendenza_T2_m * np.sqrt(sigma_pendenza_xeq_m**2 + pendenza_xeq_m**2 / pendenza_T2_m**2 * sigma_pendenza_T2_m**2)

print(f"\nAccelerazione di gravità ricavata: g = 4π² × {pendenza_xeq_m:.6f} / {pendenza_T2_m:.6f} = {g_calc:.4f} m/s²")
print(f"Incertezza accelerazione gravitazionale: {sigma_g_calc}")

print(f"\nLunghezza oscillatore se fosse ideale: L = {intercetta_xeq_m / (4 * np.pi**2):.4f} m")
 
# --------------------------------------------------------------------------
# SEZIONE G: FIT 3 - PERIODO AL QUADRATO vs POSIZIONE DI EQUILIBRIO
# --------------------------------------------------------------------------

print("\n" + "-"*80)
print("FIT 3: T² vs x_eq")
print(f"Incertezza t: {np.mean(sigma_periodi_quadri_puliti)} e x: {np.mean(sigma_x_eq_puliti)}")
print("-"*80)

risultati_fit_T2_xeq = lt.linear_fit_complete(x_eq_puliti, periodi_quadri_puliti, sigma_periodi_quadri_puliti)
lt.print_fit_results(risultati_fit_T2_xeq)

(pendenza_xeq_T2, intercetta_xeq_T2, sigma_pendenza_xeq_T2, sigma_intercetta_xeq_T2, 
 covarianza_xeq_T2, correlazione_xeq_T2, chi2_xeq_T2, chi2_ridotto_xeq_T2, 
 N_dati_xeq_T2, e_pesato_xeq_T2) = risultati_fit_T2_xeq

plt.figure(figsize=(10, 6))
lt.plot_generic_fit(
    x_eq_puliti, periodi_quadri_puliti, sigma_periodi_quadri_puliti, 
    pendenza_xeq_T2, intercetta_xeq_T2, sigma_pendenza_xeq_T2, sigma_intercetta_xeq_T2, covarianza_xeq_T2,
    title="Fit 3: T² vs Posizione di equilibrio",
    xlabel="Posizione di equilibrio $x_{eq}$ [m]",
    ylabel="Periodo al quadrato $T^2$ $[s^2]$",
    plot=True, save=False, type="Lineare"
)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("fit_lineare_T2_vs_xeq.png", dpi=300)
plt.show()

g_da_fit3 = 4 * np.pi**2 / pendenza_xeq_T2
sigma_g_da_fit3 = g_da_fit3/pendenza_xeq_T2 * sigma_pendenza_xeq_T2
print(f"\nAccelerazione di gravità ricavata direttamente:")
print(f"   g = 4π² / {pendenza_xeq_T2:.6f} = {g_da_fit3:.4f} m/s²")
print(f"Incertezza: {sigma_g_da_fit3}")

print(f"{np.mean(sigma_periodi_quadri_puliti[punti_da_rimuovere])} e {np.mean(sigma_x_eq_puliti[punti_da_rimuovere])}")

# --------------------------------------------------------------------------
# SEZIONE G.2: FIT 3 INVERSO - POSIZIONE DI EQUILIBRIO vs PERIODO AL QUADRATO
# --------------------------------------------------------------------------

print("\n" + "-"*80)
print("FIT 3 INVERSO: x_eq vs T²")
print(f"Incertezza x: {np.mean(sigma_x_eq_puliti)} e T²: {np.mean(sigma_periodi_quadri_puliti)}")
print("-"*80)

# Invertiamo le x con le y: passiamo T² come ascissa e x_eq come ordinata
# Assumiamo che sigma_x_eq_puliti sia l'incertezza sulle nuove "y"
risultati_fit_xeq_T2_inv = lt.linear_fit_complete(periodi_quadri_puliti, x_eq_puliti, sigma_x_eq_puliti)
lt.print_fit_results(risultati_fit_xeq_T2_inv)

(pendenza_inv, intercetta_inv, sigma_pendenza_inv, sigma_intercetta_inv, 
 covarianza_inv, correlazione_inv, chi2_inv, chi2_ridotto_inv, 
 N_dati_inv, e_pesato_inv) = risultati_fit_xeq_T2_inv

plt.figure(figsize=(10, 6))
lt.plot_generic_fit(
    periodi_quadri_puliti, x_eq_puliti, sigma_x_eq_puliti, 
    pendenza_inv, intercetta_inv, sigma_pendenza_inv, sigma_intercetta_inv, covarianza_inv,
    title="Fit 3: Posizione di equilibrio vs T²",
    xlabel="Periodo al quadrato $T^2$ $[s^2]$",
    ylabel="Posizione di equilibrio $x_{eq}$ [m]",
    plot=True, save=False, type="Lineare"
)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("fit_lineare_xeq_vs_T2_inv.png", dpi=300)
plt.show()

# Calcolo di g dal fit inverso: g = 4 * pi^2 * pendenza
g_da_fit3_inv = 4 * np.pi**2 * pendenza_inv
sigma_g_da_fit3_inv = 4 * np.pi**2 * sigma_pendenza_inv

print(f"\nAccelerazione di gravità ricavata dal fit inverso:")
print(f"   g = 4π² * {pendenza_inv:.6f} = {g_da_fit3_inv:.4f} m/s²")
print(f"   Incertezza su g: {sigma_g_da_fit3_inv:.4f} m/s²")

# --------------------------------------------------------------------------
# RIEPILOGO FINALE
# --------------------------------------------------------------------------

print("\n" + "="*80)
print("RIEPILOGO FINALE - RISULTATI FASE 2")
print("="*80)

print(f"\n📊 ACCELERAZIONE DI GRAVITÀ RICAVATA DA DIVERSI METODI:")
print(f"\n   Fit 3 (T² vs x_eq, DIRETTO):  g = {g_da_fit3:.2f} m/s²")
print(f"   Valore atteso (Roma):         g = 9.81 m/s²")

print(f"\n📈 DIFFERENZE:")
print(f"   Diff Fit 3: {abs(g_da_fit3 - 9.81):.3f} m/s² ({abs(g_da_fit3 - 9.81)/9.81*100:.2f}%)")

print(f"\n🎯 CHI² RIDOTTO (bontà del fit):")
print(f"   Fit 1 (T² vs m):  χ²_rid = {chi2_ridotto_T2_m:.3f}")
print(f"   Fit 2 (x_eq vs m): χ²_rid = {chi2_ridotto_xeq_m:.3f}")
print(f"   Fit 3 (T² vs x_eq): χ²_rid = {chi2_ridotto_xeq_T2:.3f}")
print(f"   (Ideale ≈ 1, <2 buono, >2 incertezze sottostimate)")

print("\n" + "="*80)
print("✅ ANALISI COMPLETATA")
print("="*80)

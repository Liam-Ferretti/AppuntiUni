import lab_tools as lt
import numpy as np
import matplotlib.pyplot as plt
"""
PRINT_SEZIONE_1 = True  
PRINT_SEZIONE_2 = True
PRINT_SEZIONE_3 = True
PRINT_SEZIONE_4 = True
"""

PRINT_SEZIONE_1 = False  
PRINT_SEZIONE_2 = False
PRINT_SEZIONE_3 = False
PRINT_SEZIONE_4 = False

RES_LUNGHEZZE = 0.001 # m
RES_TEMPO = 0.001 # s

VALORE_RIFERIMENTO = 8

PASSO = 2

SIGMA_B_LUNGHEZZE = lt.resolution_uncertainty(RES_LUNGHEZZE)
SIGMA_B_TEMPO = lt.resolution_uncertainty(RES_TEMPO)

DELTA_T = 0.0501

""" 5.2 Calibrazione del sonar """

medie_misurate = []

distanze_riferimento = lt.read_single_variable("Calibrazione/valoriref.txt")

SIGMA_A_LUNGHEZZE = []

for i in range(0, 11):
    t, x_raw = lt.read_multiple_variables(f"Calibrazione/misuraPos5_{i + 1}.txt", skip_header=2)

    analisi_campionaria_lunghezza_riferimento = lt.sample_statistic(x_raw)
    # lt.print_sample_statistics(analisi_campionaria_lunghezza_riferimento)
    SIGMA_A_LUNGHEZZE.append(analisi_campionaria_lunghezza_riferimento[3])
    
    media = np.mean(x_raw)
    medie_misurate.append(media)
    
    # Visualizzazione (opzionale, magari solo per i primi file per non intasare lo schermo)
    plt.hist(x_raw, bins=10, label="Misure grezze", alpha=0.8)
    plt.axvline(distanze_riferimento[i], color="r", label="Distanza di riferimento")
    plt.title(f"Istogramma misure sensore, x_ref = {distanze_riferimento[i]}")
    plt.xlabel("Misura")
    plt.ylabel("Frequenza")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(f"Output/istogramma_misuraPos_{distanze_riferimento[i]}.png", dpi=300, bbox_inches='tight')
    lt.showOrClose(False)


t_ref, misura_riferimento = lt.read_multiple_variables(f"Calibrazione/misuraPos5_{VALORE_RIFERIMENTO}.txt", skip_header=2)

SIGMA_A_LUNGHEZZE = max(SIGMA_A_LUNGHEZZE)

SIGMA_TOT_LUNGHEZZE = np.sqrt(SIGMA_A_LUNGHEZZE**2 + SIGMA_B_LUNGHEZZE**2)

SIGMA_TOT_TEMPO = SIGMA_B_TEMPO

print(f"Sigma totale lunghezze: {SIGMA_TOT_LUNGHEZZE} m")
print(f"Sigma totale tempo: {SIGMA_TOT_TEMPO} s")

x_sonar = np.array(medie_misurate)

x_sonar = x_sonar[1:]
distanze_riferimento = distanze_riferimento[1:]

print("Array delle medie calcolate:")
for i in range(len(x_sonar)):
    print("Mediax_sonar[i]): ", x_sonar[i], " - Distanza di riferimento: ", distanze_riferimento[i])

sigma_pos_eq = np.full_like(distanze_riferimento, SIGMA_TOT_LUNGHEZZE)

m_cal, c_cal, sig_m_cal, sig_c_cal, cov_mc_cal, rho_mc, *_ = lt.linear_fit_complete(x_sonar, distanze_riferimento, sigma_pos_eq)

lt.print_fit_results(m_cal, c_cal, sig_m_cal, sig_c_cal, cov_mc_cal, rho_mc)

lt.plot_generic_fit(x_sonar, distanze_riferimento, sigma_pos_eq, m_cal, c_cal, sig_m_cal, sig_c_cal, cov_mc_cal, title="Calibrazione del sonar", xlabel="Misura media (m)", ylabel="Distanza reale (m)", inv=True, zoom=True)

plt.savefig("Output/fit_calibrazione_sonar.png", dpi=300)
lt.showOrClose(True)

lt.scatter_pre_fit(x_sonar, distanze_riferimento, sigma_y=sigma_pos_eq, xlabel="Misura media (m)", ylabel="Distanza reale (m)", title="Dati di calibrazione del sonar")
plt.savefig("Output/scatter_calibrazione_sonar.png", dpi=300)
plt.show()

print("#" * 40 + "\n")

plt.close('all') 



"""5.3 Misura di g * µ_d"""

NUM_MISURE = 5
SIGMA_DELTA_T = np.sqrt(2) * SIGMA_TOT_TEMPO
accelerazione_list = []
SIGMA_TOT_VELOCITÀ = []
SIGMA_TOT_ACCELERAZIONI = []

SAVE_IMG = False

for i in range(0, NUM_MISURE):
    t_posizioni, posizioni = lt.read_multiple_variables(f"Theta0/misuraPosizioniMovimento_{i + 1}.txt", skip_header=2)
    indice_inizio, indice_fine = lt.find_motion_indices(t_posizioni, posizioni, 0.002, plot=PRINT_SEZIONE_2, save=SAVE_IMG, name=f"Output/indiciMovimento_{i + 1}.png")

    t_posizioni = t_posizioni[indice_inizio:indice_fine]
    posizioni = posizioni[indice_inizio:indice_fine]

    posizioni = m_cal * posizioni + c_cal

    SIGMA_TOT_LUNGHEZZE_CAL = np.sqrt(posizioni**2 * sig_m_cal**2 + m_cal**2 * SIGMA_TOT_LUNGHEZZE**2 + sig_c_cal**2)

    t_velocità, velocità = lt.numeric_derivatives(posizioni, t_posizioni, PASSO)
    t_accelerazioni, accelerazioni = lt.numeric_derivatives(velocità, t_velocità, PASSO)

    SIGMA_TOT_VELOCITÀ = lt.numeric_derivatives_uncertainty(posizioni, DELTA_T, SIGMA_TOT_LUNGHEZZE_CAL, 2 * SIGMA_DELTA_T, passo=PASSO)

    SIGMA_TOT_ACCELERAZIONI.append(lt.numeric_derivatives_uncertainty(velocità, 2 * DELTA_T, SIGMA_TOT_VELOCITÀ, 4 * SIGMA_DELTA_T, passo=PASSO))

    accelerazione_list.append(accelerazioni)

SIGMA_TOT_ACCELERAZIONI = np.concatenate(SIGMA_TOT_ACCELERAZIONI)
accelerazione_list = np.concatenate(accelerazione_list)
numeri = np.arange(len(accelerazione_list))


# Scatter plot iniziale
plt.scatter(numeri, accelerazione_list, label="Accelerazioni sperimentali")
plt.title("Accelerazioni misurate (Pre-filtraggio)")
plt.xlabel("Sequenza di misurazione")
plt.ylabel("Accelerazione [m/s^2]")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("Output/accelerazioni_misurate.png", dpi=300, bbox_inches='tight')
lt.showOrClose(PRINT_SEZIONE_2)


# Istogramma iniziale
plt.hist(accelerazione_list, bins=10, alpha=0.8, label="Accelerazioni sperimentali")
plt.title("Istogramma accelerazioni (Pre-filtraggio)")
plt.xlabel("Accelerazione [m/s^2]")
plt.ylabel("Frequenza")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("Output/istogramma_accelerazioni.png", dpi=300, bbox_inches='tight')
lt.showOrClose(PRINT_SEZIONE_2)


print(f"Misure pre-filtraggio: {len(accelerazione_list)}")


# --- FILTRAGGIO VALORI POSITIVI ---

# Usiamo una maschera per tenere solo i valori <= 0.0000001
mask = accelerazione_list <= 0.0000001
acc_filt = accelerazione_list[mask]
sigma_filt = SIGMA_TOT_ACCELERAZIONI[mask]
numeri_filt = np.arange(len(acc_filt))

print(len(accelerazione_list), len(acc_filt))

# --- CALCOLO MEDIA PESATA ---
# Il peso è l'inverso della varianza: w = 1 / sigma^2
weights = 1 / (sigma_filt**2)
acc_media_pesata = np.sum(weights * acc_filt) / np.sum(weights)

# L'incertezza della media pesata è la radice dell'inverso della somma dei pesi
sigma_acc_pesata = np.sqrt(1 / np.sum(weights))

# --- OUTPUT RISULTATI ---

print(f"Misure post-filtraggio: {len(acc_filt)}")
print(f"Accelerazione media pesata: {acc_media_pesata:.4f} ± {sigma_acc_pesata:.4f} m/s^2")

g_mu_d = -acc_media_pesata
sigma_g_mu_d = sigma_acc_pesata
print(f"RISULTATO FINALE: g * µ_d = {g_mu_d:.4f} ± {sigma_g_mu_d:.4f} m/s^2")


# --- PLOT POST-FILTRAGGIO ---
# Scatter filtrato
plt.scatter(numeri_filt, acc_filt)
plt.axhline(acc_media_pesata, color='red', linestyle='--', label='Media pesata')
plt.title("Accelerazioni misurate (Post-filtraggio)")
plt.xlabel("Sequenza di misurazione")
plt.ylabel("Accelerazione [m/s^2]")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("Output/accelerazioni_misurate.png", dpi=300, bbox_inches='tight')
lt.showOrClose(PRINT_SEZIONE_2)


# Istogramma filtrato

plt.hist(acc_filt, bins=10, alpha=0.8, label="Accelerazioni sperimentali filtrate")
plt.title("Istogramma accelerazioni (Post-filtraggio)")
plt.xlabel("Accelerazione [m/s^2]")
plt.ylabel("Frequenza")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("Output/istogramma_accelerazioni_filtrate.png", dpi=300, bbox_inches='tight')
lt.showOrClose(PRINT_SEZIONE_2)


print("\n" + "#" * 40 + "\n")

""" 5.4 Misura dell'accelerazione di gravità g """

theta = 0.02 # rad

FILE = 4

#retta di calibrazione y = m_cal * x + c_cal

#for i in range(0, 4):
t_pos_002, posizioni_misurate = lt.read_multiple_variables(f"Theta002/r/posizione_{FILE}.txt", skip_header=2)

start, end = lt.find_motion_indices(t_pos_002, posizioni_misurate, 0.005, plot=False)

posizioni_misurate = posizioni_misurate[start:end]

t_pos_002 = t_pos_002[start:end]

if PRINT_SEZIONE_3: print("File letto: ", f"Theta002/r/posizione_{i + 1}.txt")

posizioni_calibrate = m_cal * posizioni_misurate + c_cal

SIGMA_TOT_LUNGHEZZE_CAL = np.sqrt(posizioni_calibrate**2 * sig_m_cal**2 + m_cal**2 * SIGMA_TOT_LUNGHEZZE**2 + sig_c_cal**2)

plt.plot(t_pos_002, posizioni_calibrate, label="Posizione calibrata")
# plt.plot(t_pos_002, y, label="Retta di calibrazione")
plt.title("Posizione calibrata in fuzione del tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("Posizione (m)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("Output/posizione_calibrata_vs_tempo.png", dpi=300, bbox_inches='tight')
plt.show()


t_vel_002, velocità_misurate = lt.numeric_derivatives(posizioni_misurate, t_pos_002, PASSO)
t_vel_002, velocità_calibrate = lt.numeric_derivatives(posizioni_calibrate, t_pos_002, PASSO)

plt.plot(t_vel_002, velocità_calibrate, label="Velocità calibrata")
#plt.plot(t_vel_002, velocità_misurate, label="Velocità misurata", alpha=0.5)
plt.title("Velocità calibrata in fuzione del tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("Velocità (m/s)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("Output/velocità_calibrata_vs_misurata.png", dpi=300, bbox_inches='tight')
plt.show()

plt.scatter(t_pos_002, posizioni_calibrate, alpha=0.8)
plt.plot(t_pos_002, posizioni_calibrate, alpha=0.8)
plt.title("Scatter delle posizioni calibrate")
plt.xlabel("Tempo (s)")
plt.ylabel("Posizione (m)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("Output/scatter_posizioni_calibrate.png", dpi=300, bbox_inches='tight')
lt.showOrClose(False)

plt.scatter(t_vel_002, velocità_calibrate, alpha=0.8)
plt.plot(t_vel_002, velocità_calibrate, alpha=0.8)
plt.title("Scatter delle velocità calibrate")
plt.xlabel("Tempo (s)")
plt.ylabel("Velocità (m/s)")
plt.savefig("Output/scatter_velocità_calibrate.png", dpi=300, bbox_inches='tight')
lt.showOrClose(False)

SIGMA_TOT_VELOCITÀ = lt.numeric_derivatives_uncertainty(posizioni_calibrate, DELTA_T, SIGMA_TOT_LUNGHEZZE_CAL, 2 * SIGMA_DELTA_T, passo=PASSO)
# SIGMA_TOT_VELOCITÀ = SIGMA_TOT_VELOCITÀ/np.sqrt(len(SIGMA_TOT_VELOCITÀ))

m, c, sig_m, sig_c, cov_mc, rho, chi2, chi2_rid, N, weighted = lt.linear_fit_complete(t_vel_002, velocità_calibrate, SIGMA_TOT_VELOCITÀ)

lt.print_fit_results(m, c, sig_m, sig_c, cov_mc, rho, chi2, chi2_rid, N, weighted)

lt.plot_generic_fit(t_vel_002, velocità_calibrate, SIGMA_TOT_VELOCITÀ, m, c, sig_m, sig_c, cov_mc, title="Fit lineare delle velocità calibrate", xlabel="Tempo (s)", ylabel="Velocità (m/s)")
plt.savefig("Output/fit_velocità_calibrate.png", dpi=300, bbox_inches='tight')
lt.showOrClose(True)


fit_res, sigma_emp, residui = lt.linear_fit_residuals(t_vel_002, velocità_calibrate)
m, c, sig_m, sig_c, cov_mc, rho, chi2, chi2_rid, N, weighted = fit_res
lt.print_fit_results(m, c, sig_m, sig_c, cov_mc, rho, chi2, chi2_rid, N, weighted)

lt.plot_generic_fit(t_vel_002, velocità_calibrate, sigma_emp, m, c, sig_m, sig_c, cov_mc, title="Fit lineare delle velocità calibrate", xlabel="Tempo (s)", ylabel="Velocità (m/s)")
plt.savefig("Output/fit_velocità_calibrate_residui.png", dpi=300, bbox_inches='tight')
lt.showOrClose(True)


# DALLA FORMULA a = g(sin(theta) - µ_d * cos(theta)) SI RICAVA g = (a + g * µ_d * cos(theta)) / sin(theta)
# approssimando per i piccoli angoli allora:
# g = (a + g * µ_d) / theta 

g = ( m + g_mu_d ) / theta

l = 1.94
h = 0.0385

sigma_theta = np.sqrt((1/l)**2 * SIGMA_B_LUNGHEZZE**2 + (h/l**2)**2 * SIGMA_B_LUNGHEZZE**2)

print(sigma_theta)
print((m + g_mu_d)**2 / theta**2 * sigma_theta**2)

sigma_g = np.sqrt((sig_m/theta)**2 + (sigma_g_mu_d/theta)**2 + ((m + g_mu_d)/theta**2)**2 * sigma_theta**2)

print(f"Accelerazione di gravità g: {(m + g_mu_d)/theta} ± {sigma_g} m/s^2")

lt.showOrClose(False)

print("\n" + "#" * 40 + "\n")


"""5.5 Determinazione simultanea dell'accelerazione di gravità g e del coefficiente di attrito dinamico µ_d variando l'angolo theta"""

SAVE_IMG = False

thetas_per_coppia = [0.03, 0.07, 0.09, 0.01, 0.02, 0.06, 0.04, 0.08, 0.1, 0.05]
h_per_coppia = [0.0582, 0.1357, 0.1744, 0.0194, 0.0385, 0.1163, 0.0776, 0.1550, 0.1937, 0.0970]

for i in range(len(thetas_per_coppia)):
    print(thetas_per_coppia[i], np.sqrt((1/l)**2 * SIGMA_B_LUNGHEZZE**2 + (h_per_coppia[i]/l**2)**2 * SIGMA_B_LUNGHEZZE**2))

accelerazione_media_finale = []
sigma_accelerazione_finale = []

for i, theta_val in enumerate(thetas_per_coppia):
    file_idx = 2 * i + 1  # Solo il primo file di ogni coppia

    t_pos, posizioni_misurate = lt.read_multiple_variables(
        f"Angoli Variabili/r/MisuraPosizioniAngoliRipetuti_{file_idx}.txt", skip_header=2)
    print(f"File letto: MisuraPosizioniAngoliRipetuti_{file_idx}.txt (theta={theta_val} rad)")

    indice_inizio, indice_fine = lt.find_motion_indices(
        t_pos, posizioni_misurate, 0.05, 5,
        plot=False, save=SAVE_IMG,
        name=f"Output/indiciMovimento_Angolo_{theta_val}_file_{file_idx}.png")

    t_pos = t_pos[indice_inizio:indice_fine]
    posizioni_misurate = posizioni_misurate[indice_inizio:indice_fine]
    posizioni_calibrate = m_cal * posizioni_misurate + c_cal

    SIGMA_TOT_LUNGHEZZE_CAL = np.sqrt(posizioni_calibrate**2 * sig_m_cal**2 + m_cal**2 * SIGMA_TOT_LUNGHEZZE**2 + sig_c_cal**2)

    t_vel, velocità_calibrate = lt.numeric_derivatives(posizioni_calibrate, t_pos, PASSO)

    sigma_vel = lt.numeric_derivatives_uncertainty(posizioni_calibrate, DELTA_T, SIGMA_TOT_LUNGHEZZE_CAL, 2 * SIGMA_DELTA_T, passo=PASSO)

    # Fit lineare su v(t) per ottenere l'accelerazione come pendenza
    m_fit, c_fit, sig_m_fit, sig_c_fit, cov_fit, *_ = lt.linear_fit_complete(t_vel, velocità_calibrate, sigma_vel)

    accelerazione_media_finale.append(m_fit)
    sigma_accelerazione_finale.append(sig_m_fit)

    print(f"  Angolo {theta_val} rad → a = {m_fit:.4f} ± {sig_m_fit:.4f} m/s²\n")

# Ordina per angolo crescente (per il grafico)
thetas_arr = np.array(thetas_per_coppia)
h_arr = np.array(h_per_coppia)
accelerazione_arr = np.array(accelerazione_media_finale)
sigma_arr = np.array(sigma_accelerazione_finale)

idx_sort = np.argsort(thetas_arr)
thetas_arr = thetas_arr[idx_sort]
h_arr = h_arr[idx_sort]
accelerazione_arr = accelerazione_arr[idx_sort]
sigma_arr = sigma_arr[idx_sort]

# --- Fit lineare: a = g*theta - g*mu_d ---
m, q, sig_m, sig_q, cov, *_ = lt.linear_fit_complete(thetas_arr, accelerazione_arr, sigma_arr)
lt.print_fit_results(m, q, sig_m, sig_q, cov)

g_stimato = m
sig_g = sig_m
mu_d_stimato = -q / m
sig_mu_d = np.sqrt((sig_q / m)**2 + (q * sig_m / m**2)**2)

print(f"\ng stimato   : {g_stimato:.6f} ± {sig_g:.6f} m/s²")
print(f"µ_d stimato : {mu_d_stimato:.6f} ± {sig_mu_d:.6f}")

lt.plot_generic_fit(thetas_arr, accelerazione_arr, sigma_arr,
                    m, q, sig_m, sig_q, cov,
                    title="Fit lineare: accelerazione vs angolo",
                    xlabel="Angolo θ (rad)",
                    ylabel="Accelerazione (m/s²)", inv=True)
plt.savefig("Output/fit_accelerazione_vs_angolo.png", dpi=300, bbox_inches='tight')
lt.showOrClose(False)

print("\n" + "#" * 40 + "\n")


""" 5.6 Analisi dei rimbalzi del carrello """

h = 0.0970
theta = 0.05
sigma_theta = np.sqrt((1/l)**2 * SIGMA_B_LUNGHEZZE**2 + (h/l**2)**2 * SIGMA_B_LUNGHEZZE**2)

gs = []

for i in range(0, 10):
    t, pos = lt.read_multiple_variables(f"Rimbalzo/MisurePosizioneRimbalzo_{i + 1}.txt", skip_header=2)
    pos = m_cal * pos + c_cal
    i_inizio, i_stop = lt.find_motion_indices(t, pos, 0.005, plot=False)
    pos = pos[i_inizio:i_stop]
    t = t[i_inizio:i_stop]

    times_bounce, positions_bounce, types_bounce = lt.findBounce(t, pos)
    idx_peaks = [0] + [np.searchsorted(t, tb) for tb in times_bounce]

    #for j in range(0, len(idx_peaks) - 2, 2):
    for j in range(0, 1):

        # --- Prima metà (es. inizio→max oppure min→max): DISCENDENTE ---
        t_seg1   = t[idx_peaks[j]:idx_peaks[j + 1]]
        pos_seg1 = pos[idx_peaks[j]:idx_peaks[j + 1]]

        # --- Seconda metà (es. max→min): ASCENDENTE ---
        t_seg2   = t[idx_peaks[j + 1]:idx_peaks[j + 2]]
        pos_seg2 = pos[idx_peaks[j + 1]:idx_peaks[j + 2]]

        if len(t_seg1) < PASSO * 2 + 1 or len(t_seg2) < PASSO * 2 + 1:
            print(f"Segmento {j} troppo corto, skipped")
            continue

        # Accelerazione tratto discendente
        t_vel1, vel1 = lt.numeric_derivatives(pos_seg1, t_seg1, PASSO)
        sigma_vel1 = lt.numeric_derivatives_uncertainty(pos_seg1, DELTA_T, SIGMA_TOT_LUNGHEZZE, 2 * SIGMA_DELTA_T, passo=PASSO)

        vel1 = vel1[1:]
        t_vel1 = t_vel1[1:]
        sigma_vel1 = sigma_vel1[1:]

        # Fit della velocità nel tratto discendente
        m_disc, c_disc, sig_m_disc, *_ = lt.linear_fit_complete(t_vel1, vel1)
        acc_disc = m_disc
        sigma_acc_disc = sig_m_disc
        
        # Accelerazione tratto ascendente
        t_vel2, vel2 = lt.numeric_derivatives(pos_seg2, t_seg2, PASSO)
        sigma_vel2 = lt.numeric_derivatives_uncertainty(pos_seg2, DELTA_T, SIGMA_TOT_LUNGHEZZE, 2 * SIGMA_DELTA_T, passo=PASSO)

        vel2 = vel2[1:]
        t_vel2 = t_vel2[1:]
        sigma_vel2 = sigma_vel2[1:]
        
        # Fit della velocità nel tratto discendente
        m_asc, c_asc, sig_m_asc, *_ = lt.linear_fit_complete(t_vel2, vel2)
        acc_asc = m_asc
        sigma_acc_asc = sig_m_asc

        # Formula: g = (a_disc + a_asc) / (2 * theta)
        g_val   = (acc_disc + acc_asc) / (2 * theta)
        
        sum_a = acc_disc + acc_asc

        # Usando la formula corretta:
        
        sigma_g = (1 / (2 * theta)) * np.sqrt(sigma_acc_disc**2 + sigma_acc_asc**2 + (sum_a/theta)**2 * sigma_theta**2) 
        
        if j == 0:
            label = f"inizio → {types_bounce[0]} → {types_bounce[1]}"
        else:
            label = f"{types_bounce[j-1]} → {types_bounce[j]} → {types_bounce[j+1]}"

        print(f"Segmento {j//2 + 1} ({label}):")
        print(f"  a_disc  = {acc_disc:.4f} ± {sigma_acc_disc:.4f} m/s²")
        print(f"  a_asc = {acc_asc:.4f} ± {sigma_acc_asc:.4f} m/s²")
        print(f"  g      = {g_val:.4f} ± {sigma_g:.4f} m/s²")
        
        # --- Grafico dei segmenti ---
        fig, ax = plt.subplots()

        # Dati totali in grigio
        ax.scatter(t, pos, s=10, alpha=0.3, color='gray', label="Dati totali")

        # Tratto ascendente in blu
        ax.scatter(t_seg1, pos_seg1, s=20, color='blue', label=f"Discendente (seg {j//2 + 1})")

        # Tratto discendente in rosso
        ax.scatter(t_seg2, pos_seg2, s=20, color='red', label=f"Ascendente (seg {j//2 + 1})")

        # Picchi trovati da findBounce
        for ti, pi, tipo in zip(times_bounce, positions_bounce, types_bounce):
            color = 'green' if tipo == 'max' else 'orange'
            marker = '^' if tipo == 'max' else 'v'
            ax.scatter(ti, pi, color=color, marker=marker, s=150, zorder=5)

        # Evidenzia i tre punti di confine del segmento corrente
        for k in [j, j+1, j+2]:
            ax.axvline(t[idx_peaks[k]], color='black', linestyle='--', alpha=0.5)

        ax.set_title(f"File {i+1} - Segmento {j//2 + 1} ({label})")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Posizione (m)")
        ax.legend()
        plt.savefig(f"Output/segmento_file{i+1}_seg{j//2 + 1}.png", dpi=300, bbox_inches='tight')
        lt.showOrClose(False)
        plt.close()

        gs.append((g_val, sigma_g))

g_values = np.array([g for g, _ in gs])
sigma_values = np.array([s for _, s in gs])


# Media pesata
pesi = 1 / sigma_values**2
g_medio = np.sum(pesi * g_values) / np.sum(pesi)
sigma_g_medio = 1 / np.sqrt(np.sum(pesi))

print(f"g medio dai rimbalzi: {g_medio:.4f} ± {sigma_g_medio:.4f} m/s²")


# """ 5.7 Misura del coefficiente di restituzione ε """

epsilons = []
din = []
sig_din = []
dout = []
sig_dout = []

for i in range(0, 10):
    t, pos = lt.read_multiple_variables(f"Rimbalzo/MisurePosizioneRimbalzo_{i + 1}.txt", skip_header=2)
    
    # Applicazione della calibrazione
    pos = m_cal * pos + c_cal
    
    i_inizio, i_stop = lt.find_motion_indices(t, pos, 0.005, plot=False)
    pos = pos[i_inizio:i_stop]
    t   = t[i_inizio:i_stop]

    times_bounce, positions_bounce, types_bounce = lt.findBounce(t, pos)
    idx_peaks = [0] + [np.searchsorted(t, tb) for tb in times_bounce]

    if len(idx_peaks) < 3:
        print(f"File {i+1}: rimbalzi insufficienti, skipped")
        continue

    # ── Posizioni caratteristiche ──────────────────────────────────────────
    pos_start  = pos[idx_peaks[0]]   # massimo iniziale
    pos_bounce = pos[idx_peaks[1]]   # minimo al rimbalzo (≈ posizione respingente)
    pos_peak   = pos[idx_peaks[2]]   # massimo dopo il rimbalzo

    d_in  = -(pos_start  - pos_bounce)  # distanza percorsa prima dell'urto
    d_out = -(pos_peak   - pos_bounce)  # distanza percorsa dopo l'urto

    if d_in <= 0 or d_out <= 0:
        print(f"File {i+1}: distanze non valide "
              f"(d_in={d_in:.4f} m, d_out={d_out:.4f} m), skipped")
        continue

    # ── Calcolo di ε ──────────────────────────────────────────────────────
    epsilon = np.sqrt(d_out * (theta + mu_d_stimato) / (d_in * (theta - mu_d_stimato)))

    # ── Calcolo Incertezze sulle Posizioni Calibrate e sulle Distanze ─────
    
    # 1. Applichiamo la formula fornita per le singole posizioni
    sig_pos_start  = np.sqrt(pos_start**2  * sig_m_cal**2 + m_cal**2 * SIGMA_TOT_LUNGHEZZE**2 + sig_c_cal**2)
    sig_pos_bounce = np.sqrt(pos_bounce**2 * sig_m_cal**2 + m_cal**2 * SIGMA_TOT_LUNGHEZZE**2 + sig_c_cal**2)
    sig_pos_peak   = np.sqrt(pos_peak**2   * sig_m_cal**2 + m_cal**2 * SIGMA_TOT_LUNGHEZZE**2 + sig_c_cal**2)

    # 2. Propaghiamo l'incertezza sulle distanze (differenza di due posizioni => somma in quadratura)
    sigma_d_in  = np.sqrt(sig_pos_start**2 + sig_pos_bounce**2)
    sigma_d_out = np.sqrt(sig_pos_peak**2  + sig_pos_bounce**2)

    din.append(d_in)
    dout.append(d_out)
    sig_din.append(sigma_d_in)
    sig_dout.append(sigma_d_out)

    # ── Propagazione dell'incertezza su ε ─────────────────────────────────
    ratio = d_out / d_in                  
    denom = theta - mu_d_stimato          

    # Derivate parziali di ε rispetto a ciascuna variabile
    d_eps_d_dout  =  epsilon / (2 * d_out)
    d_eps_d_din   = -epsilon / (2 * d_in)
    d_eps_d_theta = -(ratio * mu_d_stimato)   / (epsilon * denom**2)
    d_eps_d_mu    =  (ratio * theta)          / (epsilon * denom**2)

    # Propagazione finale (ora utilizza le due incertezze distinte per le distanze)
    sigma_epsilon = np.sqrt(
        (d_eps_d_dout  * sigma_d_out  )**2 +
        (d_eps_d_din   * sigma_d_in   )**2 +
        (d_eps_d_theta * sigma_theta  )**2 +
        (d_eps_d_mu    * sig_mu_d     )**2
    )

    # ── Print dei risultati ───────────────────────────────────────────────
    print(f"File {i+1}:")
    print(f"  d_in  = {d_in:.4f} ± {sigma_d_in:.4f} m")
    print(f"  d_out = {d_out:.4f} ± {sigma_d_out:.4f} m")
    print(f"  ε     = {epsilon:.4f} ± {sigma_epsilon:.4f}")

    epsilons.append((epsilon, sigma_epsilon))

# ── Media pesata ───────────────────────────────────────────────────────────
epsilon_values    = np.array([e for e, _ in epsilons])
sigma_eps_values  = np.array([s for _, s in epsilons])

pesi_eps          = 1 / sigma_eps_values**2
epsilon_medio     = np.sum(pesi_eps * epsilon_values) / np.sum(pesi_eps)
sigma_eps_medio   = 1 / np.sqrt(np.sum(pesi_eps))

print(f"\nCoefficiente di restituzione medio:  ε = {epsilon_medio:.4f} ± {sigma_eps_medio:.4f}")

"""
for i in range(0, 10):
    print(f"{din[i]} pm {sig_din[i]} & {dout[i]} pm {sig_dout[i]} & {epsilon_values[i]} pm {sigma_eps_values[i]} & ")
"""
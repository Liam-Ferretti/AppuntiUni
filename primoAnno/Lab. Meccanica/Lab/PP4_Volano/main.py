import lab_tools as lt
import numpy as np
import matplotlib.pyplot as plt

RES_TEMPO = 0.00001 # secondi
SIGMA_B_TEMPO = lt.resolution_uncertainty(RES_TEMPO)

DELTA_TEMPO = 0.02 # secondi
SIGMA_TOT_DELTA_TEMPO = np.sqrt(2) * SIGMA_B_TEMPO

RES_BILANCIA = 0.0001 # kg
SIGMA_B_MASSA = lt.resolution_uncertainty(RES_BILANCIA)

RES_METRO = 0.001 # metri
SIGMA_B_METRO = lt.resolution_uncertainty(RES_METRO)

RES_NONIO = 0.00005 # metri
SIGMA_B_NONIO = lt.resolution_uncertainty(RES_NONIO) 

# 4.2 CALIBRAZIONE E VERIFICHE 
# tempo, angolo, velocità angolare, accelerazione angolare, posizione, velocità, accelerazione
t, angolo, _, _, pos, _, _ = lt.read_multiple_variables("calibrazione.csv", delimiter=";", skip_header=1)

angolo *= -1
pos *= -1

"""
plt.scatter(t, pos)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()"""

segmenti = lt.find_all_still_segments(t, pos, soglia_pos=0.0001, finestra=50, durata_minima=100, plot=False, write=False)

RES_ANGOLO = 0
RES_POSIZIONE = 0
counter_giri = 0
numero_giri = []
sigma_A_cal = []
sigma_A_sqrtN_cal = []
posizioni_cal = []

medie = []
dev_std_camp = []

for i, tupla in enumerate(segmenti): 
    posizioni_segmento = pos[tupla[0]:tupla[1]]
    angoli_segmento = angolo[tupla[0]:tupla[1]]
    tempo_segmento = t[tupla[0]:tupla[1]]
    
    if i == 0:        
        angoli_unici = np.unique(angoli_segmento)
        if len(angoli_unici) > 1:
            risoluzione_angolare = np.min(np.diff(angoli_unici))
        else:
            risoluzione_angolare = 0
        
        print("\n" + "="*40)
        print("ANALISI ANGOLO - PRIMO SEGMENTO")
        print(f"Una parte su 4000 di un angolo giro: {2 * np.pi * 1/4000:.6f}")
        print(f"Risoluzione minima (step): {risoluzione_angolare:.6f}")
        print("="*40 + "\n")

        RES_ANGOLO = risoluzione_angolare

        posizioni_uniche = np.unique(posizioni_segmento)
        if len(posizioni_uniche) > 1:
            risoluzione_posizione = np.min(np.diff(posizioni_uniche))
        else:
            risoluzione_posizione = 0
        
        
        print("\n" + "="*40)
        print("ANALISI POSIZIONE - PRIMO SEGMENTO")
        print(f"Risoluzione minima (step): {risoluzione_posizione:.8f}")
        print("="*40 + "\n")
        
        RES_POSIZIONE = risoluzione_posizione
    
    analisi_calibrazione = lt.sample_statistic(posizioni_segmento)
    valor_medio = analisi_calibrazione[0]

    sigma_A = analisi_calibrazione[2]
    sigma_A_cal.append(sigma_A)

    sigma_A_sqrtN = analisi_calibrazione[3]
    sigma_A_sqrtN_cal.append(sigma_A_sqrtN)

    numero_giri.append(counter_giri)
    posizioni_cal.append(valor_medio)

    medie.append(valor_medio)
    dev_std_camp.append(sigma_A)

    """plt.grid(True, linestyle='--', alpha=0.6)
    plt.scatter(tempo_segmento, posizioni_segmento)
    plt.plot(tempo_segmento, posizioni_segmento, label="Dati sperimentali")
    plt.axhline(valor_medio, color="r" ,linestyle="--", label=f"Valor medio: {valor_medio}")
    plt.legend()
    plt.show()"""

    counter_giri += 0.5

#elimina i primi n dati
n = 1
numero_giri = numero_giri[n:]
posizioni_cal = posizioni_cal[n:]
sigma_A_cal = sigma_A_cal[n:]
sigma_A_sqrtN_cal = sigma_A_sqrtN_cal[n:]

sigma_A_cal = np.array(sigma_A_cal)
sigma_A_sqrtN_cal = np.array(sigma_A_sqrtN_cal)

SIGMA_B_POSIZIONE = lt.resolution_uncertainty(RES_POSIZIONE)
SIGMA_B_ANGOLO = lt.resolution_uncertainty(RES_ANGOLO)

SIGMA_TOT_POSIZIONE_CAL = np.sqrt(sigma_A_sqrtN_cal**2 + SIGMA_B_POSIZIONE**2)


# Usiamo sigma_A (non divisa per sqrt(N)) perché durante il moto lavoriamo
# su singole misure di posizione, non su medie.
# Prendiamo il massimo tra i segmenti di calibrazione (valori simili tra loro)
# come stima conservativa dell'incertezza Tipo A sulla singola misura.  
sigma_A_max = max(sigma_A_cal)
print(sigma_A_max)

SIGMA_TOT_POSIZIONE_PROP = np.sqrt(sigma_A_max**2 + SIGMA_B_POSIZIONE**2)

print(f"Incertezza totale posizione: {SIGMA_TOT_POSIZIONE_PROP}")

risultati = lt.linear_fit_complete(numero_giri, posizioni_cal, SIGMA_TOT_POSIZIONE_CAL)

m, c, sig_m, sig_c, cov_mc, rho, chi2, chi2_rid, N, weighted = risultati

lt.plot_generic_fit(numero_giri, posizioni_cal, SIGMA_TOT_POSIZIONE_CAL, 
                    m, c, sig_m, sig_c, cov_mc, "Numero di giri", 
                    "Posizione", plot=False, zoom=False, save=True,
                    nome="posizione_numeroGiri")



raggiFori = lt.read_single_variable("raggifori.txt") / 1000
risultatiFori = lt.sample_statistic(raggiFori)

raggioforo = risultatiFori[0]
SIGMA_A_FORO = risultatiFori[3]

SIGMA_TOT_RAGGIO_FORO = np.sqrt(SIGMA_A_FORO**2 + SIGMA_B_NONIO**2)

raggiInterni = lt.read_single_variable("raggiinterni.txt") / 1000
risultatiInterni = lt.sample_statistic(raggiInterni)

raggioInterni = risultatiInterni[0]
SIGMA_A_Interni = risultatiInterni[3]

SIGMA_TOT_RAGGIO_INTERNO = np.sqrt(SIGMA_A_Interni**2 + SIGMA_B_NONIO**2)


spessori = lt.read_single_variable("spessori.txt") / 1000
risultatiSpessori = lt.sample_statistic(spessori)

spessore = risultatiSpessori[0]
SIGMA_A_Spessori = risultatiSpessori[3]

SIGMA_TOT_SPESSORI = np.sqrt(SIGMA_A_Spessori**2 + SIGMA_B_NONIO**2)


altezze = lt.read_single_variable("altezze.txt") / 1000
risultatiAltezze = lt.sample_statistic(altezze)

altezza = risultatiAltezze[0]
SIGMA_A_altezza = risultatiAltezze[3]

SIGMA_TOT_ALTEZZA = np.sqrt(SIGMA_A_altezza**2 + SIGMA_B_NONIO**2)


raggiRocchetto = lt.read_single_variable("raggioRocchetto.txt") / 1000
risultatiRocchetto = lt.sample_statistic(raggiRocchetto)

raggioRocchetto = risultatiRocchetto[0]
SIGMA_A_ROCCHETTO = risultatiRocchetto[3]
SIGMA_B_RAGGIO_ROCCHETTO = SIGMA_B_NONIO

SIGMA_TOT_RAGGIO_ROCCHETTO = np.sqrt(SIGMA_A_ROCCHETTO**2 + SIGMA_B_RAGGIO_ROCCHETTO**2)


C = 2 * np.pi * raggioRocchetto
SIGMA_CIRC = 2 * np.pi * SIGMA_TOT_RAGGIO_ROCCHETTO

print(C)
print(SIGMA_CIRC)

lt.plot_normal(-m, sig_m, "Circonferenza rocchetto da fit")
lt.plot_normal(C, SIGMA_CIRC, "Circonferenza rocchetto dalle misure del raggio")
plt.legend()
plt.show()


for i in range(1, len(medie)):
    print(f"numero giro: {i * 0.5}    |    Media:    {medie[i]:.10f}   |   DevStd:    {dev_std_camp[i]:.10f}")

"""
for i in range(len(posizioni_cal) - 1):
    print(posizioni_cal[i + 1] - posizioni_cal[i])
"""


# 4.3 DETERMINAZIONE MOMENTO DI INERZIA E MOMENTO DELLA FORZA D'ATTRITO
colonnePerMisura = 7
misurePerConfigurazione = 3
numeroConfigurazioni = 10

accelerazioniFit = []
configurazione = []
incertezzaAccelerazioniFit = []
quotaFit = []
incertezzaQuotaFit = []

skip = 0

for i in range(numeroConfigurazioni):
    if i == 4:
        skip = 0
    else:
        skip = 0
    t_discesa, angolo_discesa, _, _, posizione_discesa, *_ = lt.read_multiple_variables("discesaConNCoppie.csv", ";", 
                                                            skip_header = 1, 
                                                            skip_cols = i * colonnePerMisura * misurePerConfigurazione + skip * colonnePerMisura, 
                                                            n_cols = colonnePerMisura)

    # INVERSIONE DELL'ANGOLO E DELLA POSIZIONE (MOLTIPLICO PER -1)
    angolo_discesa *= -1
    posizione_discesa *= -1

    # RICERCA L'ISTANTE DI INIZIO DEL MOTO
    indice_start, _ = lt.find_motion_indices(t_discesa, posizione_discesa, soglia_pos=0.002, finestra=10, 
                                             plot=False, write=False)

    # ELIMINO I DATI PRIMA DELL'INIZIO DEL MOTO
    t_discesa = t_discesa[indice_start:]
    angolo_discesa = angolo_discesa[indice_start:]
    posizione_discesa = posizione_discesa[indice_start:]

    # PLOT DELLE POSIZIONI CON MOTO IN CORSO MA SENZA AVER TOLTO I DATI PRIMA DELL'INVERSIONE DI MOVIMENTO
    plt.scatter(t_discesa, posizione_discesa, label="Dati sperimentali posizione")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Posizione (m)")
    plt.title("Plot della posizione in funzione del tempo con moto già iniziato")
    plt.legend()
    plt.close()

    # CALCOLO DELLE VELOCITÀ DISGIUNTE CON DERIVATE NUMERICHE
    t_vel_discesa, velocità_discesa = lt.numeric_derivatives(posizione_discesa, t_discesa, 2)

    # CALCOLO DELLE INCERTEZZE SULLA VELOCITÀ
    sigma_vel_discesa = lt.numeric_derivatives_uncertainty(posizione_discesa, DELTA_TEMPO, SIGMA_TOT_POSIZIONE_PROP, SIGMA_TOT_DELTA_TEMPO, 2)

    # sigma_vel_discesa /= np.sqrt(len(sigma_vel_discesa))

    plt.errorbar(t_vel_discesa[:len(t_vel_discesa) - 10], velocità_discesa[:len(velocità_discesa) - 10], sigma_vel_discesa[:len(sigma_vel_discesa) - 10],
                 fmt="o", 
                 ms=2, 
                 label="Dati sperimentali velocità con moto corretto")
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Velocità (m/s)")
    plt.title(f"Plot della velocità completa")
    plt.legend()
    plt.close()

    # RICERCA DEGLI INDICI DI INVERSIONE DEL MOVIMENTO
    eventi = lt.find_direction_changes(t_discesa, posizione_discesa, plot=False, write=False)
    indici_tagli = np.array([e['indice'] for e in eventi])

    # ELABORAZIONE DATI PER LE SINGOLE SEZIONI DI MOTO (IN QUESTO CASO È UNA SOLA MA VERRÀ RIUSATA IN FUTURO)
    for iterazione, j in enumerate(indici_tagli):
        if iterazione == 0:
            tempo_segmento = t_discesa[0:j]
            posizione_segmento = posizione_discesa[0:j]
            angolo_segmento = angolo_discesa[0:j]
        else:
            indice_precedente = indici_tagli[iterazione - 1]
            tempo_segmento = t_discesa[indice_precedente: j]
            posizione_segmento = posizione_discesa[indice_precedente: j]
            angolo_segmento = angolo_discesa[indice_precedente: j]

        # CALCOLO DELLE VELOCITÀ DISGIUNTE CON DERIVATE NUMERICHE
        t_vel, velocità_segmento = lt.numeric_derivatives(posizione_segmento, tempo_segmento, 2)

        # CALCOLO DELLE INCERTEZZE SULLA VELOCITÀ
        sigma_vel = lt.numeric_derivatives_uncertainty(posizione_segmento, DELTA_TEMPO, SIGMA_TOT_POSIZIONE_PROP, SIGMA_TOT_DELTA_TEMPO, 2)

        # RIMOZIONE DELL'ULTIMO VALORE IN QUANTO NON È LINEARE IN OGNI MISURA
        t_vel = t_vel[:-2]
        velocità_segmento = velocità_segmento[:-2]
        sigma_vel = sigma_vel[:-2]
        
        # FIT LINEARE DELLE VELOCITÀ
        m, c, sigma_m, sigma_c, cov_mc, rho, *_ = lt.linear_fit_complete(t_vel, velocità_segmento, sigma_vel)
        # PLOT DEL FIT LINEARE
        print(len(velocità_segmento))
        lt.plot_generic_fit(t_vel, velocità_segmento, sigma_vel, 
                            m, c, sigma_m, sigma_c, cov_mc, 
                            plot=False, xlabel="Tempo (s)", ylabel="Velocità (m/s)",
                            save=True, nome=f"vel_tempo_{i}", title="Fit lineare delle velocità in funzione del tempo")

        accelerazioniFit.append(m)
        incertezzaAccelerazioniFit.append(sigma_m)
        quotaFit.append(c)
        incertezzaQuotaFit.append(sigma_c)
        configurazione.append(i + 1)
        
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        
        plt.errorbar(tempo_segmento, posizione_segmento, SIGMA_TOT_POSIZIONE_PROP, label="Dati sperimentali posizione",
                     fmt="o",
                     ms = 1)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel("Tempo (s)")
        plt.ylabel("Posizione (m)")
        plt.title(f"Plot della posizione in funzione del tempo nel range di studio: {i}")
        plt.legend()
        plt.close()

        # sigma_vel /= np.sqrt(len(sigma_vel))
        plt.errorbar(t_vel, velocità_segmento, sigma_vel, fmt="o", ms=1)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel("Tempo (s)")
        plt.ylabel("Velocità (m/s)")
        plt.title(f"Plot della velocità in funzione del tempo nel range di studio: {i}")
        plt.close()

# ── parametri (adatta ai tuoi valori) ─────────────────────────────────────────
dt        = DELTA_TEMPO
sigma_pos = SIGMA_TOT_POSIZIONE_PROP
sigma_dt  = SIGMA_TOT_DELTA_TEMPO
v_typ     = np.mean(np.abs(velocità_discesa))   # oppure un valore fisso

# ── contributi alla varianza ───────────────────────────────────────────────────
dx = v_typ * dt

var_pos  = (1 / dt)**2 * 2 * sigma_pos**2
var_time = (dx / dt**2)**2 * sigma_dt**2
var_tot  = var_pos + var_time

pct_pos  = var_pos  / var_tot * 100
pct_time = var_time / var_tot * 100
sigma_v  = np.sqrt(var_tot)

print(f"σ_v stimata      = {sigma_v:.2e} m/s")
print(f"Contributo pos   = {pct_pos:.1f}%")
print(f"Contributo tempo = {pct_time:.1f}%")

# ── torta ──────────────────────────────────────────────────────────────────────
labels  = [
    f"Posizione\n{pct_pos:.1f}%",
    f"Tempo\n{pct_time:.1f}%"
]
sizes   = [pct_pos, pct_time]
colors  = ["#3266ad", "#c45c2d"]
explode = (0.05, 0.05)

fig, ax = plt.subplots(figsize=(5, 5))
wedges, texts = ax.pie(
    sizes,
    explode   = explode,
    colors    = colors,
    labels    = labels,
    startangle= 90,
    wedgeprops= dict(linewidth=1.5, edgecolor="white"),
)
for t in texts:
    t.set_fontsize(11)

ax.set_title(
    f"Contributi alla varianza di $\\sigma_v$\n"
    f"($\\sigma_v = {sigma_v:.2e}$ m/s,  $v_{{typ}} = {v_typ:.3f}$ m/s)",
    fontsize=11, pad=14
)
plt.tight_layout()
plt.savefig("contributiVarianza.jpg", dpi=1000)
plt.close()

configurazione = np.array(configurazione)
accelerazioniFit = np.array(accelerazioniFit)
incertezzaAccelerazioniFit = np.array(incertezzaAccelerazioniFit)
quotaFit = np.array(quotaFit)
incertezzaQuotaFit = np.array(incertezzaQuotaFit)
    
for i in range(len(accelerazioniFit)):
    print(f"{i + 1} & {accelerazioniFit[i]:.8f} & {incertezzaAccelerazioniFit[i]:.8f} & {quotaFit[i]:.8f} & {incertezzaQuotaFit[i]:.8f}")

# sigma_k**2 = (1/a**2)**2 * sigma_a**2 ===> sigma_k = k**2 * sigma_a 
inversoAccelerazioni = 1/accelerazioniFit
incertezzeInverso =  inversoAccelerazioni**2 * incertezzaAccelerazioniFit

"""
for i in range(len(inversoAccelerazioni)):
    print(f"{i:2d}: | Inverso accelerazioni : {inversoAccelerazioni[i]:8.4f} | Inverso accelerazioni quadrate : {inversoAccelerazioni[i]**2:8.4f} | Incertezza inverso : {incertezzaAccelerazioniFit[i]:8.4f} | Prodotto: {incertezzeInverso[i]:8.4f}")
"""
m_inv, c_inv, sigma_m_inv, sigma_c_inv, cov_mc_inv, rho_inv, *_ = lt.linear_fit_complete(
                                                                        configurazione, 
                                                                        inversoAccelerazioni, 
                                                                        incertezzeInverso)

print(sigma_m_inv)

lt.plot_generic_fit(
    configurazione, 
    inversoAccelerazioni, 
    incertezzeInverso, 
    m_inv, 
    c_inv, 
    sigma_m_inv, 
    sigma_c_inv, 
    cov_mc_inv, 
    plot = False, 
    zoom = False,
    ylabel="Inverso delle accelerazioni",
    xlabel="Configurazione"
)

# 1. Calcolo dei valori attesi dal fit e dei residui
# y_pred = m * x + c
y_predetto = m_inv * configurazione + c_inv
residui = inversoAccelerazioni - y_predetto

# 2. Creazione della figura con due subplot
# sharex=True è fondamentale per allineare gli assi X
fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1]})
plt.subplots_adjust(hspace=0.05) # Riduce lo spazio tra i due grafici

# --- Grafico Principale (Dati + Fit) ---
ax_main.errorbar(configurazione, inversoAccelerazioni, yerr=incertezzeInverso, 
                 fmt='o', ms=5, capsize=3, label='Dati sperimentali', ecolor='black')

# Genero n punti per una linea di fit fluida
x_range = np.linspace(configurazione.min(), configurazione.max(), 200)
y_range = m_inv * x_range + c_inv
ax_main.plot(x_range, y_range, color='red', label=f'Fit: y = {m_inv:.4f}x + {c_inv:.4f}')

ax_main.set_ylabel("Inverso delle accelerazioni $[s^2/m]$")
ax_main.legend()
ax_main.grid(True, linestyle='--', alpha=0.6)
ax_main.set_title("Analisi del Fit e Residui")

# --- Grafico dei Residui ---
ax_res.errorbar(configurazione, residui, yerr=incertezzeInverso, 
                fmt='o', color='blue', ms=5, capsize=3)
ax_res.axhline(0, color='red', linestyle='--') # Linea di riferimento a zero

ax_res.set_xlabel("Configurazione")
ax_res.set_ylabel("Residui")
ax_res.grid(True, linestyle='--', alpha=0.6)

plt.savefig("fit_invAccelerazioniConResidui")
plt.close()

g = 9.803

numBulloni = 20
masseBulloni = []
sigma_A_massa_bulloni = []

print(f"Sigma B: {SIGMA_B_MASSA}")

for i in range(numBulloni):
    massaBullone = lt.read_multiple_variables("masseBulloni.txt", skip_cols = i, n_cols = 1)[0] / 1000
    
    risultati_masse = lt.sample_statistic(massaBullone)

    masseBulloni.append(risultati_masse[0])
    sigma_A_massa_bulloni.append(risultati_masse[2])

sigma_A_massa_bulloni = np.array(sigma_A_massa_bulloni)

SIGMA_TOT_MASSA_BULLONI = np.sqrt(sigma_A_massa_bulloni**2 + SIGMA_B_MASSA**2)

pesi = 1/SIGMA_TOT_MASSA_BULLONI
massaBullone = np.average(masseBulloni, weights=pesi)

SIGMA_TOT_MASSA_BULLONE = max(SIGMA_TOT_MASSA_BULLONI)

raggiBulloni_raw = lt.read_single_variable("raggioBullone.txt") / 1000

raggiBullone = raggiBulloni_raw + raggioRocchetto

risultatiBulloni = lt.sample_statistic(raggiBullone)

raggioBullone = risultatiBulloni[0]
SIGMA_A_RAGGIO_BULLONE = risultatiBulloni[3]
SIGMA_TOT_RAGGIO_BULLONE = np.sqrt(SIGMA_A_RAGGIO_BULLONE**2 + SIGMA_B_METRO**2)

SIGMA_TOT_RAGGIO_BULLONE = np.sqrt(SIGMA_TOT_RAGGIO_BULLONE**2 + SIGMA_TOT_RAGGIO_ROCCHETTO**2)


raggiDisco_raw = lt.read_single_variable("raggioDisco.txt") / 1000
raggiDisco = raggiDisco_raw + raggioRocchetto

risultatiDisco = lt.sample_statistic(raggiDisco)

raggioDisco = risultatiDisco[0]
SIGMA_A_RAGGIO_DISCO = risultatiDisco[3]
SIGMA_TOT_RAGGIO_DISCO = np.sqrt(SIGMA_A_RAGGIO_DISCO**2 + SIGMA_B_METRO**2)

SIGMA_TOT_RAGGIO_DISCO = np.sqrt(SIGMA_TOT_RAGGIO_DISCO**2 + SIGMA_TOT_RAGGIO_ROCCHETTO**2)

massaMassa = lt.read_single_variable("massaMassa.txt") / 1000
risultatiMassaMassa = lt.sample_statistic(massaMassa)

massaMassa = risultatiMassaMassa[0]
SIGMA_A_MASSA_MASSA = risultatiMassaMassa[3]
SIGMA_TOT_MASSA_MASSA = np.sqrt(SIGMA_A_MASSA_MASSA**2 + SIGMA_B_MASSA**2)

print(f"Valor medio: {massaMassa} | SIGMA_A: {SIGMA_A_MASSA_MASSA} | SIGMA_TOT_MASSA_BULLONE: {SIGMA_TOT_MASSA_MASSA}")


print("="*50)

for i in range(len(raggiRocchetto)):
    print(f"{i:2d} | {raggiRocchetto[i]}")

print(f"Valor medio: {raggioRocchetto} +/- {SIGMA_TOT_RAGGIO_ROCCHETTO}")
print(f"Sigma A: {SIGMA_A_ROCCHETTO}")
print(f"Sigma B: {SIGMA_B_RAGGIO_ROCCHETTO}")
print("="*50)

for i in range(len(raggiBullone)):
    print(f"{i:2d} | {raggiBullone[i]}")

print(f"Valor medio: {raggioBullone} +/- {SIGMA_TOT_RAGGIO_BULLONE}")
print(f"Sigma A: {SIGMA_A_RAGGIO_BULLONE}")
print(f"Sigma B: {SIGMA_B_METRO}")
print("="*50)

for i in range(len(raggiDisco)):
    print(f"{i:2d} | {raggiDisco[i]}")

print(f"Valor medio: {raggioDisco} +/- {SIGMA_TOT_RAGGIO_DISCO}")
print(f"Sigma A: {SIGMA_A_RAGGIO_DISCO}")
print(f"Sigma B: {SIGMA_B_METRO}")
print("="*50)

print("raggiFori")
for i, j in enumerate(raggiFori):
    print(f"{i:2d} | {j}")

print(f"Valor medio: {raggioforo} +/- {SIGMA_TOT_RAGGIO_FORO}")
print(f"Sigma A: {SIGMA_A_FORO}")
print(f"Sigma B: {SIGMA_B_NONIO}")
print("="*50)


print("raggi intern")
for i, j in enumerate(raggiInterni):
    print(f"{i:2d} | {j}")

print(f"Valor medio: {raggioInterni} +/- {SIGMA_TOT_RAGGIO_INTERNO}")
print(f"Sigma A: {SIGMA_A_Interni}")
print(f"Sigma B: {SIGMA_B_NONIO}")
print("="*50)


print("spessori")
for i, j in enumerate(spessori):
    print(f"{i:2d} | {j}")

print(f"Valor medio: {spessore} +/- {SIGMA_TOT_SPESSORI}")
print(f"Sigma A: {SIGMA_A_Spessori}")
print(f"Sigma B: {SIGMA_B_NONIO}")
print("="*50)


print("altezze")
for i, j in enumerate(altezze):
    print(f"{i:2d} | {j}")

print(f"Valor medio: {altezza} +/- {SIGMA_TOT_SPESSORI}")
print(f"Sigma A: {SIGMA_A_altezza}")
print(f"Sigma B: {SIGMA_B_NONIO}")
print("="*50)


I0 = 2 * massaBullone * raggioBullone**2 * c_inv/m_inv - massaMassa * raggioRocchetto**2

SIGMA_I0 = np.sqrt((2 * raggioBullone**2 * (c_inv/m_inv))**2 * SIGMA_TOT_MASSA_BULLONE**2 + 
                   (4 * massaBullone * raggioBullone * (c_inv/m_inv))**2 * SIGMA_TOT_RAGGIO_BULLONE**2 +
                   (2 * massaBullone * raggioBullone**2 * (1/m_inv))**2 * sigma_c_inv**2 + 
                   (2 * massaBullone * raggioBullone**2 * (c_inv/m_inv**2))**2 * sigma_m_inv**2 +
                   (raggioRocchetto**2)**2 * SIGMA_TOT_MASSA_MASSA**2 + 
                   (2 * raggioRocchetto * massaMassa)**2 * SIGMA_TOT_RAGGIO_ROCCHETTO**2 -
                   (2 * cov_mc_inv * (2 * massaBullone * raggioBullone**2 * (1/m_inv)) * (2 * massaBullone * raggioBullone**2 * (c_inv/m_inv**2)))
                )

Ma = massaMassa * g * raggioRocchetto - (2 * massaBullone * raggioBullone**2) / (raggioRocchetto * m_inv)

SIGMA_Ma = np.sqrt(
    (g*raggioRocchetto)**2 * SIGMA_TOT_MASSA_MASSA**2 +
    (massaMassa * g + (2 * massaBullone * raggioBullone**2)/(m_inv * raggioRocchetto**2))**2 * SIGMA_TOT_RAGGIO_ROCCHETTO**2 + 
    ((2 * raggioBullone**2)/(m_inv * raggioRocchetto))**2 * SIGMA_TOT_MASSA_BULLONE**2 +
    ((4 * massaBullone * raggioBullone)/(m_inv * raggioRocchetto))**2 * SIGMA_TOT_RAGGIO_BULLONE**2 +
    ((2 * massaBullone * raggioBullone**2)/(m_inv**2 * raggioRocchetto))**2 * sigma_m_inv**2
)

print(f"Momento di inerzia del disco: {I0:.6e} +/- {SIGMA_I0:.6e} Kg * m^2")
print(f"Momento d'attrito Ma: {Ma:.6e} +/- {SIGMA_Ma:.6e} Nm")

print("Calcolo teorico di I0")

rho = 2700

I0_teo = np.pi * rho/2 * (raggioDisco**4 * spessore + altezza * (raggioInterni**4 - raggioRocchetto**4) - 
                          20* raggioforo**4 * spessore - 40* raggioforo**2 * spessore * raggioBullone**2)

C = np.pi * rho / 2

dI_draggioDisco = C * (4 * raggioDisco**3 * spessore)
dI_dspessore    = C * (raggioDisco**4 - 20 * raggioforo**4 - 40 * raggioforo**2 * raggioBullone**2)
dI_daltezza     = C * (raggioInterni**4 - raggioRocchetto**4)
dI_draggioInt   = C * (4 * altezza * raggioInterni**3)
dI_draggioRoc   = C * (-4 * altezza * raggioRocchetto**3)
dI_draggioforo  = C * (-80 * raggioforo**3 * spessore - 80 * raggioforo * spessore * raggioBullone**2)
dI_draggioBul   = C * (-80 * raggioforo**2 * spessore * raggioBullone)

sigma_I0_teorico = np.sqrt(
    (dI_draggioDisco * SIGMA_TOT_RAGGIO_DISCO)**2 +
    (dI_dspessore    * SIGMA_TOT_SPESSORI)**2 +
    (dI_daltezza     * SIGMA_TOT_ALTEZZA)**2 +
    (dI_draggioInt   * SIGMA_TOT_RAGGIO_INTERNO)**2 +
    (dI_draggioRoc   * SIGMA_TOT_RAGGIO_ROCCHETTO)**2 +
    (dI_draggioforo  * SIGMA_TOT_RAGGIO_FORO)**2 +
    (dI_draggioBul   * SIGMA_TOT_RAGGIO_BULLONE)**2
)

print(f"Momento di inerzia del disco: {I0_teo:.6e} +/- {sigma_I0_teorico:.6e} Kg * m^2")

plt.figure(figsize=(12,6))
lt.plot_normal(I0_teo, sigma_I0_teorico, r"$I_0$ teorico")
lt.plot_normal(I0, SIGMA_I0, r"$I_0$ ricavato dal fit")
plt.xlabel(r"$Kg \cdot m^2$")
plt.ylabel("Frequenza")
plt.title(r"Confronto tra i valori di $I_0$")
plt.legend()
plt.savefig("confrontoI0.jpg", dpi = 1000)
plt.show()

# 4.4
numeroMisureAttrito = 4
valoriFinaliDaRimuovere = 20

velocità_limite = []
incertezze_velocità_limite = []

for i in range(numeroMisureAttrito):
    # print(f"Misura {i + 1}")
    tempo_attrito, angolo_attrito, _, _, posizione_attrito, *_ = lt.read_multiple_variables("misuraAttrito.csv", 
                                                            delimiter=";",
                                                            skip_header = 1, 
                                                            skip_cols = i * colonnePerMisura, 
                                                            n_cols = colonnePerMisura)
    
    angolo_attrito *= -1
    posizione_attrito *= -1

    start, stop = lt.find_motion_indices(tempo_attrito, posizione_attrito, soglia_pos=0.001, finestra=10, 
                                         plot=False, write=False)
    
    stop = len(tempo_attrito) - 50

    tempo_attrito = tempo_attrito[start:]
    angolo_attrito = angolo_attrito[start:]
    posizione_attrito = posizione_attrito[start:]

    sig_pos = np.full_like(tempo_attrito, SIGMA_TOT_POSIZIONE_PROP)

    t_vel_attrito, velocità_attrito = lt.numeric_derivatives(posizione_attrito, tempo_attrito, 2)
    sigma_vel_attrito = lt.numeric_derivatives_uncertainty(posizione_attrito, DELTA_TEMPO, sig_pos, SIGMA_TOT_DELTA_TEMPO, 2)
    
    plt.errorbar(t_vel_attrito, velocità_attrito, 
        sigma_vel_attrito,         # Array delle incertezze
        fmt='o',                   # 'o' mantiene l'aspetto simile a scatter
        markersize=3,              # Dimensione del punto (equivalente a s=10)
        capsize=3,                 # Aggiunge le "alette" orizzontali alle barre
        ecolor='gray',              # Colore delle barre d'errore (opzionale)
        elinewidth=1,              # Spessore delle linee d'errore
        label=f"Velocità dell'intera misurazione (m/s)")

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.close()

    # RICERCA DEGLI INDICI DI INVERSIONE DEL MOVIMENTO
    eventi_attrito = lt.find_direction_changes(tempo_attrito, posizione_attrito, 0.005, 
                                               plot=False, write=False)
    
    indici_tagli = np.array([e['indice'] for e in eventi_attrito])

    # ELABORAZIONE DATI PER LE SINGOLE SEZIONI DI MOTO (IN QUESTO CASO È UNA SOLA MA VERRÀ RIUSATA IN FUTURO)
    if indici_tagli.size > 0:
        j = indici_tagli[0]

        tempo_attrito_segmento = tempo_attrito[0:j]
        angolo_attrito_segmento = angolo_attrito[0:j]
        posizione_attrito_segmento = posizione_attrito[0:j]
        
    i0 = 14 * 50 - start
    i1 = 24 * 50 - start

    # Taglia i dati al range lineare
    tempo_attrito_lin   = tempo_attrito_segmento[i0 : i1+1]
    posizione_attrito_lin   = posizione_attrito_segmento[i0 : i1+1]
    sig_lin = np.full_like(tempo_attrito_lin, SIGMA_TOT_POSIZIONE_PROP)

    # FIT LINEARE DELLE POSIZIONI IN CONDIZIONE DI LINEARTÀ PER STIMARE LA VELOCITÀ LIMITE
    m_lim, c_lim, sigma_m_lim, sigma_c_lim, cov_mc_lim, rho_lim, *_ = lt.linear_fit_complete(tempo_attrito_lin, posizione_attrito_lin, sig_lin)

    lt.plot_generic_fit(tempo_attrito_lin, posizione_attrito_lin, sig_lin, m_lim, c_lim, sigma_m_lim, sigma_c_lim, cov_mc_lim, plot=False)

    # DERIVATE NUMERICHE DELLE VELOCITÀ
    t_vel_attrito_lin, velocità_attrito_lin = lt.numeric_derivatives(posizione_attrito_lin, tempo_attrito_lin, 2)

    sigma_velocità_attrito_lin = lt.numeric_derivatives_uncertainty(posizione_attrito_lin, DELTA_TEMPO, 
                                                                    SIGMA_TOT_POSIZIONE_PROP, SIGMA_TOT_DELTA_TEMPO, 2)
    
    pesi = 1/sigma_velocità_attrito_lin**2
    media_vel_attrito_lin = np.average(velocità_attrito_lin, weights=pesi)

    sigma_media_pesata_vel_lin = np.sqrt(1 / np.sum(pesi))

    print(f"Velocità limite (da fit):                {m_lim} +- {sigma_m_lim}")
    print(f"Velocità limite (da derivate numeriche): {np.mean(media_vel_attrito_lin)} +- {sigma_media_pesata_vel_lin}")

    velocità_limite.append(media_vel_attrito_lin)
    incertezze_velocità_limite.append(sigma_media_pesata_vel_lin)


    # 4.5 
    # 1. Calcola la velocità su tutto il segmento
    t_vel_segmento, v_segmento = lt.numeric_derivatives(posizione_attrito_segmento, tempo_attrito_segmento, 2)
    sigma_vel_segmento = lt.numeric_derivatives_uncertainty(posizione_attrito_segmento, DELTA_TEMPO, SIGMA_TOT_POSIZIONE_PROP, SIGMA_TOT_DELTA_TEMPO)
    
    plt.scatter(t_vel_segmento, v_segmento)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Velocità (m)")
    plt.title("Plot della velocità in funzione del tempo")
    plt.close()

    # 2. Definisci la soglia (95% della velocità limite calcolata dal fit)
    soglia_v = 0.80 * m_lim

    # 3. Trova il primo indice che supera la soglia
    # np.argmax su una maschera booleana restituisce il PRIMO indice 'True'
    # indici_oltre_soglia = np.where(t_vel_segmento >= 5)[0]
    indici_oltre_soglia = np.where(v_segmento >= soglia_v)[0]

    if indici_oltre_soglia.size > 0:
        idx_taglio = indici_oltre_soglia[0]
    else:
        # Se non raggiunge mai il 95%, prendi tutto (o gestisci l'eccezione)
        idx_taglio = len(v_segmento)

    # 4. Taglia drasticamente i dati: teniamo solo [0 : idx_taglio]
    t_transiente = t_vel_segmento[:idx_taglio]
    v_transiente = v_segmento[:idx_taglio]
    sigma_transiente = sigma_vel_segmento[:idx_taglio]

    # 5. Calcolo del logaritmo solo sui dati "puliti" del transiente
    # Usiamo m_lim come asintoto
    y_arg = (m_lim - v_transiente) / m_lim       # argomento (non il log)
    y_log = np.log(y_arg)                          # il log

    sigma_z = (sigma_transiente / m_lim) / y_arg  # = sigma_v / (m_lim - v)

    sigma_arg = 1/m_lim * np.sqrt(sigma_transiente**2 + v_transiente**2/m_lim**2 * sigma_m_lim**2)

    t_taglio = t_vel_segmento[idx_taglio]

    plt.figure(figsize=(10, 5))
    plt.scatter(tempo_attrito_segmento, posizione_attrito_segmento, s=2, color='steelblue')

    plt.axvspan(t_taglio, tempo_attrito_segmento[-1],
                color='gray', alpha=0.2, label=f"Zona esclusa (v ≥ {soglia_v:.3f} m/s)")

    plt.xlabel("Tempo (s)")
    plt.ylabel("Posizione (m)")
    plt.title(f"Posizione nel tempo - Misura {i+1}")
    plt.grid(True, ls='--')
    plt.legend()
    plt.close()

    # 6. Plot del transiente linearizzato
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        t_transiente, 
        y_arg, 
        yerr=sigma_arg,            # Array delle incertezze
        fmt='o',                   # 'o' mantiene l'aspetto simile a scatter
        markersize=3,              # Dimensione del punto (equivalente a s=10)
        capsize=3,                 # Aggiunge le "alette" orizzontali alle barre
        ecolor='red',              # Colore delle barre d'errore (opzionale)
        elinewidth=1,              # Spessore delle linee d'errore
        label=f"Dati troncati per velocità maggiori di {soglia_v:.3f} (m/s)"
    )
    plt.xlabel("Tempo (s)")
    plt.ylabel(r"$(v_{lim} - v)/v_{lim}$")
    plt.title(f"Analisi Semilog Transiente - Misura {i+1}")
    plt.grid(True, ls='--')
    plt.legend()
    plt.yscale("log")
    plt.savefig(f"ScatterSemilog_{i}.jpg", dpi=1000)
    plt.close()

    risultati = lt.linear_fit_complete(t_transiente, y_log, sigma_z)
    
    m, c, sig_m, sig_c, cov_mc, *_ = risultati
    lt.plot_generic_fit(t_transiente, y_log, sigma_z, m, c, sig_m, sig_c, cov_mc, "Tempo (s)", r"$\ln((v_{lim} - v)/v_{lim})$", plot=False, save=True, nome=f"semilog_{i + 1}")




velocità_limite = np.array(velocità_limite)
incertezze_velocità_limite = np.array(incertezze_velocità_limite)

pesi = 1/incertezze_velocità_limite**2
v_limite_media = np.average(velocità_limite, weights=pesi)
incertezza_vel_limite = np.sqrt(1 / np.sum(pesi))

k = (massaMassa * g * raggioRocchetto - Ma) / (v_limite_media * raggioRocchetto)

SIGMA_K = np.sqrt(((g * raggioRocchetto) / (v_limite_media * raggioRocchetto))**2 * SIGMA_TOT_MASSA_MASSA**2 +
                  ((g * massaMassa) / (v_limite_media * raggioRocchetto))**2 * SIGMA_TOT_RAGGIO_ROCCHETTO**2 +
                  ((Ma) / (v_limite_media * raggioRocchetto))**2 * SIGMA_Ma**2 + 
                  ((massaMassa * g * raggioRocchetto - Ma) / (v_limite_media**2 * raggioRocchetto))**2 * incertezza_vel_limite**2 + 
                  ((massaMassa * g * raggioRocchetto - Ma) / (v_limite_media * raggioRocchetto**2))**2 * SIGMA_TOT_RAGGIO_ROCCHETTO**2
                  )

print(f"Velocità limite {v_limite_media:.6e} +/- {incertezza_vel_limite:.6e}")
print(f"Coefficente della velocità nell'attrito viscoso: {k:.6e} +/- {SIGMA_K:.6e}")



# 4.6
rapporti_totali = []
ma_totali = []  # <-- lista per raccogliere i valori di Ma
sigma_rapporti_totali = []
sigma_ma_totali = [] 

for i in range(4):
    misuraDaLeggere = i
    t_deltah, _, _, _, posizione_deltah, *_ = lt.read_multiple_variables("dh.csv", delimiter=";", skip_header=1, 
                                                                        skip_cols=colonnePerMisura * misuraDaLeggere,
                                                                        n_cols=colonnePerMisura)
    posizione_deltah *= -1

    h_max = max(posizione_deltah)
    
    posizione_deltah = posizione_deltah - h_max
    
    start, _ = lt.find_motion_indices(t_deltah, posizione_deltah, 0.0002, finestra=15, inizio_ricerca=0.2, plot=False)
    
    h_iniziale = posizione_deltah[0]
    
    t_deltah = t_deltah[start:]
    
    posizione_deltah = posizione_deltah[start:]

    lt.scatter_pre_fit(t_deltah, posizione_deltah, SIGMA_TOT_POSIZIONE_PROP, xlabel="Tempo (s)", ylabel="Posizione (m)", title="Scatter delle posizioni in funzione del tempo")
    plt.savefig(f"scatterPosizioneRimbalzi_{i}.jpg", dpi=1000)
    plt.close()
    
    eventi_deltah = lt.find_direction_changes(t_deltah, posizione_deltah, prominenza=0.03, distanza=10, plot=False, write=False)
    
    altezze_picchi = [e['posizione'] for e in eventi_deltah if e['tipo'] == 'min']
    altezze_picchi.insert(0, h_iniziale)

    rapporti_dissipazione = []
    
    ma_values = []

    for j in range(3, 8):
        h_pre = altezze_picchi[j - 1]
        h_post = altezze_picchi[j]
        
        delta_h = h_pre - h_post
        dh_over_h = delta_h / h_pre                            # Deltah/h (adimensionale)
        rapporto = dh_over_h                                   # in percentuale
        rapporti_dissipazione.append(rapporto)

        sigma_delta_h = SIGMA_TOT_POSIZIONE_PROP/h_pre * np.sqrt(1 + (h_post/h_pre)**2)
        sigma_rapporti_totali.append(sigma_delta_h)

        # Calcolo Ma
        Ma_val = (massaMassa * g * raggioRocchetto * dh_over_h) / (2 - dh_over_h)
        ma_values.append(Ma_val)

        sigma_ma_val = np.sqrt(((g * raggioRocchetto * dh_over_h) / (2 - dh_over_h))**2 * SIGMA_TOT_MASSA_MASSA +
                               ((massaMassa * g * dh_over_h) / (2 - dh_over_h))**2 * SIGMA_TOT_RAGGIO_ROCCHETTO + 
                               ((2 * massaMassa * g * raggioRocchetto ) / (2 - dh_over_h)**2)**2 * sigma_delta_h)
        
        sigma_ma_totali.append(sigma_ma_val)

    rapporti_totali.append(rapporti_dissipazione)
    ma_totali.append(ma_values)  # <-- accumulo
    punti = np.arange(len(rapporti_dissipazione))

# --- Analisi statistica DeltaH/H (invariata) ---
rapporti_totali = np.array(rapporti_totali)
rapporti_totali = np.concatenate(rapporti_totali)

sigma_rapporti_totali = np.array(sigma_rapporti_totali)
sigma_ma_totali = np.array(sigma_ma_totali)

for i in rapporti_totali:
    print(i)

risultati_delta_h = lt.sample_statistic(rapporti_totali)

sigma_delta_h_best = np.sqrt(1 / np.sum(1/sigma_rapporti_totali**2))

print(f"Sigma DH/H: {sigma_delta_h_best}")

plt.figure(figsize=(12, 8))
lt.plot_normal(risultati_delta_h[0], risultati_delta_h[2])
plt.hist(rapporti_totali, bins=5, label="Dati sperimentali ", density=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("DeltaH/H")
plt.ylabel("Frequenza")
plt.title(f"Istrogramma della frequenza del deltaH/H")
plt.legend()
plt.savefig("istoDeltaH.jpg", dpi = 1000)
plt.show()

lt.print_sample_statistics(risultati_delta_h)
print(f"Valor medio dei rapporti delle altezze: {risultati_delta_h[0]} +/- {risultati_delta_h[3]}")

# --- Analisi statistica Ma ---
ma_totali = np.concatenate(ma_totali)

print("Ma")
for i, j in enumerate(ma_totali):
    print(f"{i:2d}: {j}")

risultati_ma = lt.sample_statistic(ma_totali)

sigma_ma_totale_best = np.sqrt(1 / np.sum(1 / sigma_ma_totali**2))

print(f"Sigma MA: {sigma_ma_totale_best}")

plt.figure(figsize=(12, 8))
lt.plot_normal(risultati_ma[0], risultati_ma[2])
plt.hist(ma_totali, label="Dati sperimentali Ma", density=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Ma")
plt.ylabel("Frequenza")
plt.title("Istogramma della frequenza di Ma")
plt.legend()
plt.savefig("istoFrequenzeMa.jpg", dpi = 1000)
plt.show()

lt.print_sample_statistics(risultati_ma)
print(f"Valor medio di Ma: {risultati_ma[0]} +/- {risultati_ma[3]}")

# SI NOTA CHE IL VALORE DEL DH/H NON È COSTANTE AL VARIARE DELLE ALTEZZE, CIÒ PUÒ TROVARE UNA SPIEGAZIONE NELLA NON INESTENSIBILITÀ
# DEL FILO IN QUANTO PIÙ LA MASSA SCENDE VELOCEMENTE PIÙ ESSE HA ENERGIA PER ESTENDERE IL FILO PRIMA DI INVERTIRE IL MOTO
# (NON SO SE ABBIA EFFETTIVAMENTE SENSO)


# confronto tra i due Ma trovati:

plt.figure(figsize=(13, 8))
lt.plot_normal(Ma, SIGMA_Ma, label="Momento della forza tramite fit lineare")
lt.plot_normal(risultati_ma[0], risultati_ma[3], label=r"Momento della forza tramite $\frac{\Delta H}{H}$")
plt.ylabel("Frequenza")
plt.xlabel(r"$M_a$ $[N \cdot m]$")
plt.legend()
plt.savefig("plotConfrontoMa.jpg", dpi = 1000)
plt.show()
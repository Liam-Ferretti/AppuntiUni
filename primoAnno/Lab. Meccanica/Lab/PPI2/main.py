import numpy as np
import matplotlib.pyplot as plt
import lab_tools as lt

RES_NONIO = 0.05 # mm
RES_BILANCIA = 0.1 # mm
RES_CRONOMETRO = 0.01 # s

SIGMA_NONIO = lt.resolution_uncertainty(RES_NONIO)
SIGMA_BILANCIA = lt.resolution_uncertainty(RES_BILANCIA)
SIGMA_CRONOMETRO = lt.resolution_uncertainty(RES_CRONOMETRO)

print("\n" + "="*50 + "\n")

print(f"Sigma Nonio: {SIGMA_NONIO:.6f} mm")
print(f"Sigma Bilancia: {SIGMA_BILANCIA:.6f} mm")
print(f"Sigma Cronometro: {SIGMA_CRONOMETRO:.6f} mm")

print("\n" + "="*50 + "\n")

# ANALISI DISTANZA R

misureDiametroPerno = lt.read_single_variable("DiametroPerno.txt")
risultatiDiametroPerno = lt.sample_statistic(misureDiametroPerno)
valoreDiametroPerno = risultatiDiametroPerno[0]
SIGMA_A_DIAMETRO_PERNO = risultatiDiametroPerno[3]
SIGMA_TOT_DIAMETRO_PERNO = np.sqrt(SIGMA_A_DIAMETRO_PERNO**2 + SIGMA_NONIO**2)
print(f"Diametro Perno: {valoreDiametroPerno:.6f} +- {SIGMA_TOT_DIAMETRO_PERNO:.6f} mm")

misureDistanzeForo = lt.read_single_variable("misureDistanzaFori.txt")
risultatiDistanzeForo = lt.sample_statistic(misureDistanzeForo)
valoreDistanzeForo = risultatiDistanzeForo[0]
SIGMA_A_DISTANZE_FORO = risultatiDistanzeForo[3]
SIGMA_TOT_DISTANZE_FORO = np.sqrt(SIGMA_A_DISTANZE_FORO**2 + SIGMA_NONIO**2)
print(f"Distanza Foro: {valoreDistanzeForo:.6f} +- {SIGMA_TOT_DISTANZE_FORO:.6f} mm")

misureDiametroForiBulloni = lt.read_single_variable("diametroBulloni.txt")
risultatiDiametroForiBulloni = lt.sample_statistic(misureDiametroForiBulloni)
valoreDiametroForiBulloni = risultatiDiametroForiBulloni[0]
SIGMA_A_DIAMETRO_FORI_BULLONI = risultatiDiametroForiBulloni[3]
SIGMA_TOT_DIAMETRO_FORI_BULLONI = np.sqrt(SIGMA_A_DIAMETRO_FORI_BULLONI**2 + SIGMA_NONIO**2)
print(f"Diametro Fori Bulloni: {valoreDiametroForiBulloni:.6f} +- {SIGMA_TOT_DIAMETRO_FORI_BULLONI:.6f} mm")

distanzaR = valoreDiametroPerno/2 + valoreDistanzeForo + valoreDiametroForiBulloni/2
SIGMA_R = np.sqrt((0.5*SIGMA_TOT_DIAMETRO_PERNO)**2 + SIGMA_TOT_DISTANZE_FORO**2 + (0.5*SIGMA_TOT_DIAMETRO_FORI_BULLONI)**2)
print(f"Distanza R: {distanzaR:.6f} +- {SIGMA_R:.6f} mm")

# Converti R in metri per i calcoli di k e I0
R_m = distanzaR * 1e-3          # m
SIGMA_R_m = SIGMA_R * 1e-3      # m

print("\n" + "="*50 + "\n")
# ANALISI MASSA M

misureMassaM = lt.read_single_variable("MassaBulloni.txt")
risultatiMassaM = lt.sample_statistic(misureMassaM)
valoreMassaM = risultatiMassaM[0]
SIGMA_A_MASSA_M = risultatiMassaM[3]
SIGMA_TOT_MASSA_M = np.sqrt(SIGMA_A_MASSA_M**2 + SIGMA_BILANCIA**2)
print(f"Massa M: {valoreMassaM:.6f} +- {SIGMA_TOT_MASSA_M:.6f} g")

# Converti massa in kg per i calcoli di k e I0
valoreMassaM_kg = valoreMassaM * 1e-3       # kg
SIGMA_TOT_MASSA_M_kg = SIGMA_TOT_MASSA_M * 1e-3  # kg

# ANALISI PERIODI IN FUNZIONE DEL NUMERO DI BULLONI (TRAMITE MISURA DI 5 OSCILLAZIONI COMPLETE):

print("\n" + "="*50 + "\n")
print("Analisi Periodi in funzione del numero di bulloni:")

nBulloni = np.array([0, 2, 3, 4, 6])

arr_periodo = np.zeros(len(nBulloni))
arr_SIGMA_PERIODO = np.zeros(len(nBulloni))
arr_periodoQuadro = np.zeros(len(nBulloni))
arr_SIGMA_PERIODO_QUADRO = np.zeros(len(nBulloni))

for i in range(len(nBulloni)):
    misureTempoTotale = lt.read_single_variable(f"tempi5Oscillazioni{nBulloni[i]}Bulloni.txt")
    risultatiTempoTotale = lt.sample_statistic(misureTempoTotale)
    valoreTempoTotale = risultatiTempoTotale[0]
    SIGMA_A_TEMPO_TOTALE = risultatiTempoTotale[3]
    SIGMA_TOT_TEMPO_TOTALE = np.sqrt(SIGMA_A_TEMPO_TOTALE**2 + SIGMA_CRONOMETRO**2)

    periodo = valoreTempoTotale / 5
    SIGMA_PERIODO = SIGMA_TOT_TEMPO_TOTALE / 5

    arr_periodo[i] = periodo
    arr_SIGMA_PERIODO[i] = SIGMA_PERIODO

    periodoQuadro = periodo**2
    SIGMA_PERIODO_QUADRO = 2 * periodo * SIGMA_PERIODO

    arr_periodoQuadro[i] = periodoQuadro
    arr_SIGMA_PERIODO_QUADRO[i] = SIGMA_PERIODO_QUADRO

for i, (periodo, SIGMA_PERIODO) in enumerate(zip(arr_periodo, arr_SIGMA_PERIODO)):
    print(f"Periodo con {nBulloni[i]} bulloni: {periodo:.6f} +- {SIGMA_PERIODO:.6f} s")

for i, (periodoQuadro, SIGMA_PERIODO_QUADRO) in enumerate(zip(arr_periodoQuadro, arr_SIGMA_PERIODO_QUADRO)):
    print(f"Periodo^2 con {nBulloni[i]} bulloni: {periodoQuadro:.6f} +- {SIGMA_PERIODO_QUADRO:.6f} s^2")

print("\n" + "="*50 + "\n")
print("ANALISI FIT LINEARE PESATO: T^2 vs M_B")

# X = massa totale bulloni in kg (unita' SI per k in N*m/rad e I0 in kg*m^2)
X = nBulloni * valoreMassaM_kg   # kg
Y = arr_periodoQuadro            # s^2
sigma_Y = arr_SIGMA_PERIODO_QUADRO

W = 1.0 / (sigma_Y ** 2)
sum_W = np.sum(W)

x_mean = np.sum(W * X) / sum_W
y_mean = np.sum(W * Y) / sum_W
x2_mean = np.sum(W * (X**2)) / sum_W
xy_mean = np.sum(W * X * Y) / sum_W

var_x = x2_mean - (x_mean**2)
cov_xy = xy_mean - (x_mean * y_mean)

m = cov_xy / var_x
c = y_mean - m * x_mean

delta = (sum_W**2) * var_x

sigma_m = np.sqrt(sum_W / delta)
sigma_c = np.sqrt(np.sum(W * (X**2)) / delta)
cov_mc = - np.sum(W * X) / delta
rho = cov_mc / (sigma_m * sigma_c)

# Incertezze medie pesate (Tabella 5)
sigma_x_mean = 0.0
sigma_x2_mean = 0.0
sigma_var_x = 0.0
sigma_y_mean = 1.0 / np.sqrt(sum_W)
sigma_xy_mean = np.sqrt(x2_mean / sum_W)
sigma_cov_xy = np.sqrt(var_x / sum_W)

print("\n--- DATI PER TABELLA 5 (Input del Fit - sigma_X trascurabile) ---")
print(f"x_bar (Media pesata M_B):   {x_mean:.6f} +- {sigma_x_mean:.6f} kg")
print(f"y_bar (Media pesata T^2):   {y_mean:.6f} +- {sigma_y_mean:.6f} s^2")
print(f"x2_bar:                     {x2_mean:.6f} +- {sigma_x2_mean:.6f} kg^2")
print(f"xy_bar:                     {xy_mean:.6f} +- {sigma_xy_mean:.6f} kg*s^2")
print(f"Var[x] pesata:              {var_x:.6f} +- {sigma_var_x:.6f} kg^2")
print(f"Cov[x,y] pesata:            {cov_xy:.6f} +- {sigma_cov_xy:.6f} kg*s^2")
print(f"Somma dei pesi (sigma^-2):  {sum_W:.6f} s^-4")

print("\n--- DATI PER TABELLA 6 (Risultati del Fit) ---")
print(f"Coefficiente angolare (m):  {m:.6f} +- {sigma_m:.6f} s^2/kg")
print(f"Intercetta (c):             {c:.6f} +- {sigma_c:.6f} s^2")
print(f"Covarianza [m,c]:           {cov_mc:.6e} s^4/kg")
print(f"Coeff. di correlazione rho: {rho:.6f}")

print("\n" + "="*50 + "\n")
print("ANALISI DEI RESIDUI E CHI QUADRO")

y_fit = m * X + c
residui = Y - y_fit
residui_normalizzati = residui / sigma_Y

chi_quadro = np.sum(residui_normalizzati**2)
dof = len(X) - 2
chi_quadro_ridotto = chi_quadro / dof

print(f"Chi-quadro sperimentale:     {chi_quadro:.4f}")
print(f"Gradi di libertà (dof):      {dof}")
print(f"Chi-quadro ridotto:          {chi_quadro_ridotto:.4f}")

for i in range(len(nBulloni)):
    print(f"Configurazione {nBulloni[i]} bulloni -> Residuo: {residui[i]:.6f} s^2 | Normalizzato: {residui_normalizzati[i]:.2f}")
    

# =====================================================================
# GRAFICO: DATI + FIT + RESIDUI (con punto estrapolato evidenziato)
# =====================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7),
                               gridspec_kw={'height_ratios': [3, 1]})

# Pannello superiore
ax1.errorbar(X*1e3, Y, yerr=sigma_Y, fmt='o', color='darkblue', ecolor='red',
             capsize=4, label='Dati sperimentali ($T^2$)')

x_plot = np.linspace(-5e-3, max(X)*1.15, 200)
ax1.plot(x_plot*1e3, m * x_plot + c, color='green', linestyle='--',
         label=f'Fit lineare: $m={m:.4f}$ s²/kg, $c={c:.4f}$ s²')

ax1.set_ylabel(r'$T^2$ [s²]', fontsize=11)
ax1.set_title(r'$T^2$ vs $M_B$: fit lineare', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=9)

# Pannello inferiore: residui normalizzati (solo punti del fit)
ax2.errorbar(X*1e3, residui_normalizzati, yerr=np.ones_like(X), fmt='o',
             color='purple', ecolor='purple', capsize=4, label='Residui normalizzati')
ax2.axhline(0, color='black', linestyle='-', linewidth=1)
ax2.axhline(2,  color='red', linestyle=':', alpha=0.5, label=r'$\pm 2\sigma$')
ax2.axhline(-2, color='red', linestyle=':', alpha=0.5)

ax2.set_xlabel(r'$M_B$ [g]', fontsize=11)
ax2.set_ylabel(r'Residui [$\delta/\sigma$]', fontsize=11)
ax2.set_ylim(-3.5, 3.5)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig("fitLineare.png", dpi=300)
plt.show()

# =====================================================================
# PUNTO 5: Calcolo di k e I0 con propagazione degli errori (inclusa covarianza m,c)
# ---------------------------------------------------------------------
# Dalla relazione T^2 = (4*pi^2/k)*I0 + (4*pi^2*R^2/k)*M_B si ha:
#   m = 4*pi^2 * R^2 / k   =>  k  = 4*pi^2 * R^2 / m
#   c = 4*pi^2 * I0 / k    =>  I0 = c * k / (4*pi^2) = c * R^2 / m
# =====================================================================

print("\n" + "="*50 + "\n")
print("PUNTO 5: Calcolo di k e I0 dai parametri del fit")

k = 4 * np.pi**2 * R_m**2 / m        # N*m/rad
I0 = c * R_m**2 / m                  # kg*m^2

# --- Propagazione errori per k = 4*pi^2 * R^2 / m ---
# d(k)/dR = 8*pi^2 * R / m
# d(k)/dm = -4*pi^2 * R^2 / m^2 = -k/m
# Nessun termine misto (R e m indipendenti)
sigma_k = k * np.sqrt((2 * SIGMA_R_m / R_m)**2 + (sigma_m / m)**2)

# --- Propagazione errori per I0 = c * R^2 / m ---
# d(I0)/dc  =  R^2 / m        =>  contributo: (sigma_c/c)^2 * I0^2 / ... -> relativo: sigma_c/c
# d(I0)/dm  = -c * R^2 / m^2  =>  contributo relativo: sigma_m/m
# d(I0)/dR  =  2*c*R / m      =>  contributo relativo: 2*sigma_R/R
# Termine di covarianza tra m e c (NON trascurabile perché rho != 0):
#   2 * (dI0/dm)*(dI0/dc) * cov_mc / I0^2 = 2 * (-1/m)*(1/c) * cov_mc
#                                           = -2 * cov_mc / (m * c)
var_I0_rel = (
    (sigma_c / c)**2
    + (sigma_m / m)**2
    + (2 * SIGMA_R_m / R_m)**2
    - 2 * cov_mc / (m * c)      # termine covarianza: segno - perché dI0/dm < 0
)
sigma_I0 = abs(I0) * np.sqrt(var_I0_rel)

print(f"\nCostante di torsione:   k  = {k:.6e} +- {sigma_k:.6e} N*m/rad")
print(f"Momento di inerzia:     I0 = {I0:.6e} +- {sigma_I0:.6e} kg*m^2")

# Compatibilità relativa (numero di sigma)
print(f"\nIncertezza relativa k:  {sigma_k/k*100:.2f} %")
print(f"Incertezza relativa I0: {sigma_I0/abs(I0)*100:.2f} %")

# =====================================================================
# PARTE II: Estrapolazione per configurazione con doppi dadi (6 bulloni, 2 dadi ciascuno)
# =====================================================================

print("\n" + "="*50 + "\n")
print("PARTE II: Estrapolazione per configurazione con doppi dadi")

misureTempoTotale_DD = lt.read_single_variable("tempi5Oscillazioni6BulloniDoppiDadi.txt")
risultatiTempoTotale_DD = lt.sample_statistic(misureTempoTotale_DD)
valoreTempoTotale_DD = risultatiTempoTotale_DD[0]
SIGMA_A_TEMPO_TOTALE_DD = risultatiTempoTotale_DD[3]
SIGMA_TOT_TEMPO_TOTALE_DD = np.sqrt(SIGMA_A_TEMPO_TOTALE_DD**2 + SIGMA_CRONOMETRO**2)

periodo_DD = valoreTempoTotale_DD / 5
SIGMA_periodo_DD = SIGMA_TOT_TEMPO_TOTALE_DD / 5
periodoQuadro_DD = periodo_DD**2
SIGMA_periodoQuadro_DD = 2 * periodo_DD * SIGMA_periodo_DD

print(f"Periodo misurato (doppi dadi): {periodo_DD:.6f} +- {SIGMA_periodo_DD:.6f} s")
print(f"T^2 misurato:                  {periodoQuadro_DD:.6f} +- {SIGMA_periodoQuadro_DD:.6f} s^2")

# Massa della configurazione doppi dadi: serve la massa di ciascun bullone con 2 dadi
# (letta da file separato; se identica a valoreMassaM_kg usa quella)
# Assumiamo che il file "MassaBulloniDoppiDadi.txt" contenga le misure della massa
# del singolo bullone con doppio dado; altrimenti sostituire con il valore corretto
misureMassaDD = lt.read_single_variable("MassaBulloniDoppiDadi.txt")
risultatiMassaDD = lt.sample_statistic(misureMassaDD)
MB_DD = risultatiMassaDD[0] * 1e-3                                          # kg
SIGMA_MB_DD = np.sqrt(risultatiMassaDD[3]**2 + SIGMA_BILANCIA**2) * 1e-3   # kg

# Estrapolazione da retta di fit: T^2_est = m * MB_DD + c
T2_est = m * MB_DD + c

# Propagazione incertezza sull'estrapolazione:
# sigma_T2_est = sqrt( (dT2/dm)^2 * sigma_m^2 + (dT2/dc)^2 * sigma_c^2
#                     + 2*(dT2/dm)*(dT2/dc)*cov_mc )
# = sqrt( MB_DD^2 * sigma_m^2 + sigma_c^2 + 2*MB_DD*cov_mc )
SIGMA_T2_est = np.sqrt(
    MB_DD**2 * sigma_m**2
    + sigma_c**2
    + 2 * MB_DD * cov_mc
)

T_est = np.sqrt(T2_est)
SIGMA_T_est = SIGMA_T2_est / (2 * T_est)

print(f"\nMassa configurazione doppi dadi (M_B): {MB_DD*1e3:.4f} +- {SIGMA_MB_DD*1e3:.4f} g")
print(f"T^2 estrapolato:  {T2_est:.6f} +- {SIGMA_T2_est:.6f} s^2")
print(f"T   estrapolato:  {T_est:.6f} +- {SIGMA_T_est:.6f} s")

# Compatibilità tra valore estrapolato e misurato
discrepanza_T2 = abs(periodoQuadro_DD - T2_est)
sigma_discrepanza_T2 = np.sqrt(SIGMA_periodoQuadro_DD**2 + SIGMA_T2_est**2)
n_sigma = discrepanza_T2 / sigma_discrepanza_T2

print(f"\nDiscrepanza T^2: {discrepanza_T2:.6f} s^2")
print(f"Incertezza combinata: {sigma_discrepanza_T2:.6f} s^2")
print(f"Compatibilità: {n_sigma:.2f} sigma")
if n_sigma < 2:
    print("=> I valori sono COMPATIBILI (< 2 sigma)")
elif n_sigma < 3:
    print("=> Compatibilità marginale (2-3 sigma)")
else:
    print("=> I valori NON sono compatibili (> 3 sigma)")

# =====================================================================
# GRAFICO: DATI + FIT + RESIDUI (con punto estrapolato evidenziato)
# =====================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7),
                               gridspec_kw={'height_ratios': [3, 1]})

# Pannello superiore
ax1.errorbar(X*1e3, Y, yerr=sigma_Y, fmt='o', color='darkblue', ecolor='red',
             capsize=4, label='Dati sperimentali ($T^2$)')

x_plot = np.linspace(-5e-3, max(X)*1.15, 200)
ax1.plot(x_plot*1e3, m * x_plot + c, color='green', linestyle='--',
         label=f'Fit lineare: $m={m:.4f}$ s²/kg, $c={c:.4f}$ s²')

# Punto estrapolato Parte II
ax1.errorbar(MB_DD*1e3, T2_est, yerr=SIGMA_T2_est, fmt='s', color='orange',
             ecolor='orange', capsize=5, markersize=8, label='Estrapolato (doppi dadi)')
ax1.errorbar(MB_DD*1e3, periodoQuadro_DD, yerr=SIGMA_periodoQuadro_DD, fmt='^',
             color='red', ecolor='red', capsize=5, markersize=8, label='Misurato (doppi dadi)')

ax1.set_ylabel(r'$T^2$ [s²]', fontsize=11)
ax1.set_title(r'$T^2$ vs $M_B$: fit lineare e confronto configurazione doppi dadi', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=9)

# Pannello inferiore: residui normalizzati (solo punti del fit)
ax2.errorbar(X*1e3, residui_normalizzati, yerr=np.ones_like(X), fmt='o',
             color='purple', ecolor='purple', capsize=4, label='Residui normalizzati')
ax2.axhline(0, color='black', linestyle='-', linewidth=1)
ax2.axhline(2,  color='red', linestyle=':', alpha=0.5, label=r'$\pm 2\sigma$')
ax2.axhline(-2, color='red', linestyle=':', alpha=0.5)

ax2.set_xlabel(r'$M_B$ [g]', fontsize=11)
ax2.set_ylabel(r'Residui [$\delta/\sigma$]', fontsize=11)
ax2.set_ylim(-3.5, 3.5)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig("figura.png", dpi=300)
plt.show()


# =====================================================================
# BONUS: Stima teorica di I_disco e confronto con I0 dal fit
# ---------------------------------------------------------------------
# Il disco di alluminio ha N_FORI fori cilindrici di raggio r_b a distanza R
# dal centro. Il momento di inerzia corretto è:
#
#   I_teo = I_disco_pieno - N_FORI * I_foro
#
# con:
#   I_disco_pieno = (1/2) * M_d * r_d^2
#   I_foro        = (1/2) * m_foro * r_b^2 + m_foro * R^2   (teorema assi paralleli)
#   m_foro        = rho_Al * pi * r_b^2 * h_d
#
# Grandezze misurate:
#   - Diametro del disco  (D_d): metro/squadra   -> DiametroDisco.txt   [mm]
#   - Spessore del disco  (h_d): calibro a nonio -> SpessoreDisco.txt   [mm]
#   - Diametro fori bulloni (D_b): già misurato  -> diametroBulloni.txt [mm]
#   - R: già calcolato sopra
#   - Massa del disco: letta da MassaDisco.txt se disponibile,
#     altrimenti calcolata da geometria + densità alluminio.
# =====================================================================
 
print("\n" + "="*50 + "\n")
print("BONUS: Stima teorica di I_disco (con sottrazione fori) e confronto con I0")
 
RHO_AL  = 2700.0   # kg/m^3, densità alluminio
N_FORI  = 6        # numero di fori per i bulloni
 
# --- Diametro del disco ---
misureDiametroDisco = lt.read_single_variable("DiametroDisco.txt")
risultatiDiametroDisco = lt.sample_statistic(misureDiametroDisco)
valoreDiametroDisco = risultatiDiametroDisco[0]          # mm
SIGMA_A_DiamDisco = risultatiDiametroDisco[3]
RES_METRO = 1.0    # mm  (metro/squadra)
SIGMA_B_METRO = lt.resolution_uncertainty(RES_METRO)
SIGMA_TOT_DIAMETRO_DISCO = np.sqrt(SIGMA_A_DiamDisco**2 + SIGMA_B_METRO**2)
print(f"Diametro disco:  {valoreDiametroDisco:.3f} +- {SIGMA_TOT_DIAMETRO_DISCO:.3f} mm")
 
r_d       = (valoreDiametroDisco / 2) * 1e-3             # m
sigma_r_d = (SIGMA_TOT_DIAMETRO_DISCO / 2) * 1e-3        # m
 
# --- Spessore del disco ---
misureSpessoreDisco = lt.read_single_variable("SpessoreDisco.txt")
risultatiSpessoreDisco = lt.sample_statistic(misureSpessoreDisco)
valoreSpessoreDisco = risultatiSpessoreDisco[0]          # mm
SIGMA_A_Spess = risultatiSpessoreDisco[3]
SIGMA_TOT_SPESSORE_DISCO = np.sqrt(SIGMA_A_Spess**2 + SIGMA_NONIO**2)
print(f"Spessore disco:  {valoreSpessoreDisco:.3f} +- {SIGMA_TOT_SPESSORE_DISCO:.3f} mm")
 
h_d       = valoreSpessoreDisco * 1e-3                   # m
sigma_h_d = SIGMA_TOT_SPESSORE_DISCO * 1e-3              # m
 
# --- Raggio dei fori (già misurato nella Parte I) ---
r_b       = (valoreDiametroForiBulloni / 2) * 1e-3       # m
sigma_r_b = (SIGMA_TOT_DIAMETRO_FORI_BULLONI / 2) * 1e-3 # m
 
# --- Massa del disco ---
try:
    misureMassaDisco = lt.read_single_variable("MassaDisco.txt")
    risultatiMassaDisco = lt.sample_statistic(misureMassaDisco)
    valoreMassaDisco = risultatiMassaDisco[0]              # g
    SIGMA_A_MD = risultatiMassaDisco[3]
    SIGMA_TOT_MASSA_DISCO = np.sqrt(SIGMA_A_MD**2 + SIGMA_BILANCIA**2)
    M_d       = valoreMassaDisco * 1e-3                    # kg
    sigma_M_d = SIGMA_TOT_MASSA_DISCO * 1e-3               # kg
    print(f"Massa disco (pesata): {valoreMassaDisco:.3f} +- {SIGMA_TOT_MASSA_DISCO:.3f} g")
except Exception:
    # Fallback: massa del DISCO PIENO dalla geometria (sovrastima, verrà corretta)
    M_d_pieno = RHO_AL * np.pi * r_d**2 * h_d
    sigma_M_d_pieno = M_d_pieno * np.sqrt((2*sigma_r_d/r_d)**2 + (sigma_h_d/h_d)**2)
    # Massa rimossa dai fori
    M_fori_tot = N_FORI * RHO_AL * np.pi * r_b**2 * h_d
    M_d = M_d_pieno - M_fori_tot
    # Propagazione (r_b e h_d contribuiscono anche qui, ma M_d_pieno domina)
    sigma_M_fori_tot = M_fori_tot * np.sqrt((2*sigma_r_b/r_b)**2 + (sigma_h_d/h_d)**2)
    sigma_M_d = np.sqrt(sigma_M_d_pieno**2 + sigma_M_fori_tot**2)
    print(f"Massa disco (da geometria, corretta per fori): {M_d*1e3:.3f} +- {sigma_M_d*1e3:.3f} g")
 
# -----------------------------------------------------------------------
# MOMENTO DI INERZIA DEL DISCO PIENO
# -----------------------------------------------------------------------
I_pieno = 0.5 * M_d * r_d**2
# Propagazione relativa: (sigma_M/M)^2 + (2*sigma_r/r)^2
sigma_I_pieno = I_pieno * np.sqrt((sigma_M_d / M_d)**2 + (2 * sigma_r_d / r_d)**2)
 
# -----------------------------------------------------------------------
# MOMENTO DI INERZIA DI UN SINGOLO FORO (cilindretto rimosso)
# Teorema assi paralleli: I_foro = (1/2)*m_foro*r_b^2 + m_foro*R^2
# con m_foro = rho_Al * pi * r_b^2 * h_d
# -----------------------------------------------------------------------
m_foro = RHO_AL * np.pi * r_b**2 * h_d
# Propagazione di m_foro: sigma_m/m = sqrt((2*sigma_rb/rb)^2 + (sigma_h/h)^2)
sigma_m_foro = m_foro * np.sqrt((2 * sigma_r_b / r_b)**2 + (sigma_h_d / h_d)**2)
 
I_foro_spin  = 0.5 * m_foro * r_b**2           # asse proprio del foro
I_foro_paral = m_foro * R_m**2                  # termine di Steiner
I_foro       = I_foro_spin + I_foro_paral
 
# Propagazione di I_foro:
# dI/dm_foro = (1/2)*r_b^2 + R^2
# dI/dr_b    = m_foro * r_b
# dI/dR      = 2 * m_foro * R
dI_dm = 0.5 * r_b**2 + R_m**2
dI_drb = m_foro * r_b
dI_dR  = 2 * m_foro * R_m
sigma_I_foro = np.sqrt(
    (dI_dm * sigma_m_foro)**2
    + (dI_drb * sigma_r_b)**2
    + (dI_dR  * SIGMA_R_m)**2
)
 
# -----------------------------------------------------------------------
# MOMENTO DI INERZIA TEORICO CORRETTO
# I_teo = I_pieno - N_FORI * I_foro
# -----------------------------------------------------------------------
I_teo       = I_pieno - N_FORI * I_foro
sigma_I_teo = np.sqrt(sigma_I_pieno**2 + (N_FORI * sigma_I_foro)**2)
 
print(f"\nI_disco pieno:      {I_pieno:.4e} +- {sigma_I_pieno:.4e} kg*m^2")
print(f"I_foro (singolo):   {I_foro:.4e}  +- {sigma_I_foro:.4e}  kg*m^2")
print(f"  di cui I_spin:    {I_foro_spin:.4e} kg*m^2")
print(f"  di cui I_Steiner: {I_foro_paral:.4e} kg*m^2")
print(f"N_fori * I_foro:    {N_FORI*I_foro:.4e} kg*m^2")
print(f"\nI_teo (corretto):   {I_teo:.4e} +- {sigma_I_teo:.4e} kg*m^2")
print(f"I0 dal fit:         {I0:.4e} +- {sigma_I0:.4e} kg*m^2")
 
# --- Compatibilità (z-score) ---
discrepanza_I = abs(I0 - I_teo)
sigma_discrepanza_I = np.sqrt(sigma_I0**2 + sigma_I_teo**2)
z_I = discrepanza_I / sigma_discrepanza_I
 
print(f"\nDiscrepanza:          {discrepanza_I:.4e} kg*m^2")
print(f"Incertezza combinata: {sigma_discrepanza_I:.4e} kg*m^2")
print(f"z-score:              {z_I:.2f} sigma")
if z_I < 2:
    print("=> I_teo e I0 dal fit sono COMPATIBILI (< 2 sigma)")
elif z_I < 3:
    print("=> Compatibilità marginale (2-3 sigma)")
else:
    print("=> I valori NON sono compatibili (> 3 sigma)")
 
print(f"\nIncertezza relativa I_teo: {sigma_I_teo/I_teo*100:.2f} %")
print(f"Incertezza relativa I0:    {sigma_I0/abs(I0)*100:.2f} %")

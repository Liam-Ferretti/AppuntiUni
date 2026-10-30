import lab_tools as lt
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# PARAMETRI INIZIALI
# ==========================================
RES_CRONOMETRO = 0.01

NUM_OSCILLAZIONI = 20

sigma_b_cronometro = lt.resolution_uncertainty(RES_CRONOMETRO)
print(f"Incertezza di risoluzione del cronometro (sui tempi totali): {sigma_b_cronometro:.5f} s\n")
print(f"Incertezza di risoluzione del cronometro (sul periodo): {sigma_b_cronometro/NUM_OSCILLAZIONI:.6f} s\n")

# ==========================================
# PRIMO SET DI MISURE (MASSA 1)
# ==========================================
print("--- ANALISI MASSA 1 ---")

# 1. Leggiamo i tempi totali GREZZI
tempi20misureMassa1 = lt.read_single_variable('misurePeriodo1.txt')

# 2. Facciamo l'analisi statistica sui tempi grezzi
ris_stat_tempi_1 = lt.sample_statistic(tempi20misureMassa1)


# 3. Calcoliamo l'incertezza totale sui tempi grezzi
ris_fin_tempi_1 = lt.get_total_uncertainty(ris_stat_tempi_1[0], ris_stat_tempi_1[3], sigma_b_cronometro)

# 4. DIVIDIAMO IL RISULTATO FINALE PER IL NUMERO DI OSCILLAZIONI
periodo_medio_1 = ris_fin_tempi_1[0] / NUM_OSCILLAZIONI
incertezza_totale_periodo_1 = ris_fin_tempi_1[3] / NUM_OSCILLAZIONI

print(f'Risultati sul PERIODO (Massa 1):')
print(f'Periodo medio: {periodo_medio_1:.6f} s')
print(f'Dev standard medio: {ris_stat_tempi_1[2]/NUM_OSCILLAZIONI:.6f} s')
print(f'Sigma A: {ris_stat_tempi_1[3]/NUM_OSCILLAZIONI:.6f} s')
print(f'Incertezza totale: {incertezza_totale_periodo_1:.6f} s\n')

# 5. Grafico (dobbiamo dividere l'array dei tempi per plottare l'istogramma dei periodi)
periodi_massa1 = tempi20misureMassa1 / NUM_OSCILLAZIONI

plt.hist(periodi_massa1, bins=5, edgecolor='black', density=True, label='Dati Sperimentali 1')
lt.plot_normal(periodo_medio_1, ris_stat_tempi_1[2]/NUM_OSCILLAZIONI)
plt.xlabel('Periodo (s)')
plt.ylabel('Frequenza relativa')
plt.title('Istogramma dei periodi di oscillazione (Massa 1)')
plt.legend()
plt.savefig('istogramma_periodi_massa1.png')
plt.show()


# ==========================================
# SECONDO SET DI MISURE (MASSA 2)
# ==========================================
print("--- ANALISI MASSA 2 ---")

# 1. Leggiamo i tempi totali GREZZI
tempi20misureMassa2 = lt.read_single_variable('misurePeriodo2.txt')

# 2. Facciamo l'analisi statistica sui tempi grezzi
ris_stat_tempi_2 = lt.sample_statistic(tempi20misureMassa2)

# 3. Calcoliamo l'incertezza totale sui tempi grezzi
ris_fin_tempi_2 = lt.get_total_uncertainty(ris_stat_tempi_2[0], ris_stat_tempi_2[3], sigma_b_cronometro)

# 4. DIVIDIAMO IL RISULTATO FINALE PER IL NUMERO DI OSCILLAZIONI
periodo_medio_2 = ris_fin_tempi_2[0] / NUM_OSCILLAZIONI
incertezza_totale_periodo_2 = ris_fin_tempi_2[3] / NUM_OSCILLAZIONI


print(f'Risultati sul PERIODO (Massa 2):')
print(f'Periodo medio: {periodo_medio_2:.6f} s')
print(f'Dev standard medio: {ris_stat_tempi_2[2]/NUM_OSCILLAZIONI:.6f} s')
print(f'Sigma A: {ris_stat_tempi_2[3]/NUM_OSCILLAZIONI:.6f} s')
print(f'Incertezza totale: {incertezza_totale_periodo_2:.6f} s\n')

# 5. Grafico 
periodi_massa2 = tempi20misureMassa2 / NUM_OSCILLAZIONI

plt.hist(periodi_massa2, bins=5, edgecolor='black', density=True, label='Dati Sperimentali 2')
lt.plot_normal(periodo_medio_2, ris_stat_tempi_2[2]/NUM_OSCILLAZIONI)
plt.xlabel('Periodo (s)')
plt.ylabel('Frequenza relativa')
plt.title('Istogramma dei periodi di oscillazione (Massa 2)')
plt.legend()
plt.savefig('istogramma_periodi_massa2.png')
plt.show()



tempo1, tempo2, tempo3, tempo4 = lt.read_multiple_variables('misurePeriodi.txt')

risultati_tempo1 = lt.sample_statistic(tempo1)
risultati_tempo2 = lt.sample_statistic(tempo2)
risultati_tempo3 = lt.sample_statistic(tempo3)
risultati_tempo4 = lt.sample_statistic(tempo4)

risultati_tempo1_fin = lt.get_total_uncertainty(risultati_tempo1[0], risultati_tempo1[3], sigma_b_cronometro)
risultati_tempo2_fin = lt.get_total_uncertainty(risultati_tempo2[0], risultati_tempo2[3], sigma_b_cronometro)
risultati_tempo3_fin = lt.get_total_uncertainty(risultati_tempo3[0], risultati_tempo3[3], sigma_b_cronometro)
risultati_tempo4_fin = lt.get_total_uncertainty(risultati_tempo4[0], risultati_tempo4[3], sigma_b_cronometro)

risultati_periodo1_fin = np.array(risultati_tempo1_fin) / NUM_OSCILLAZIONI
risultati_periodo2_fin = np.array(risultati_tempo2_fin) / NUM_OSCILLAZIONI
risultati_periodo3_fin = np.array(risultati_tempo3_fin) / NUM_OSCILLAZIONI
risultati_periodo4_fin = np.array(risultati_tempo4_fin) / NUM_OSCILLAZIONI

print(f"Periodo 1: {risultati_periodo1_fin[0]:.6f} s")
print(f"Periodo^2 1: {risultati_periodo1_fin[0]**2:.6f} s^2")
print(f"Dev std 1: {risultati_periodo1_fin[2]:.6f} s")
print(f"Sigma A 1: {risultati_periodo1_fin[3]:.6f} s")
print(f"Incertezza totale 1: {risultati_periodo1_fin[3]:.6f} s")
print(f"Incertezza totale periodo quadro: {2*risultati_periodo1_fin[0]*risultati_periodo1_fin[3]:.6f} s^2\n")

print(f"Periodo 2: {risultati_periodo2_fin[0]:.6f} s")
print(f"Periodo^2 2: {risultati_periodo2_fin[0]**2:.6f} s^2")
print(f"Dev std 2: {risultati_periodo2_fin[2]:.6f} s")
print(f"Sigma A 2: {risultati_periodo2_fin[3]:.6f} s")
print(f"Incertezza totale 2: {risultati_periodo2_fin[3]:.6f} s")
print(f"Incertezza totale periodo quadro: {2*risultati_periodo2_fin[0]*risultati_periodo2_fin[3]:.6f} s^2\n")

print(f"Periodo 3: {risultati_periodo3_fin[0]:.6f} s")
print(f"Periodo^2 3: {risultati_periodo3_fin[0]**2:.6f} s^2")
print(f"Dev std 3: {risultati_periodo3_fin[2]:.6f} s")
print(f"Sigma A 3: {risultati_periodo3_fin[3]:.6f} s")
print(f"Incertezza totale 3: {risultati_periodo3_fin[3]:.6f} s")
print(f"Incertezza totale periodo quadro: {2*risultati_periodo3_fin[0]*risultati_periodo3_fin[3]:.6f} s^2\n")

print(f"Periodo 4: {risultati_periodo4_fin[0]:.6f} s")
print(f"Periodo^2 4: {risultati_periodo4_fin[0]**2:.6f} s^2")
print(f"Dev std 4: {risultati_periodo4_fin[2]:.6f} s")
print(f"Sigma A 4: {risultati_periodo4_fin[3]:.6f} s")
print(f"Incertezza totale 4: {risultati_periodo4_fin[3]:.6f} s")
print(f"Incertezza totale periodo quadro: {2*risultati_periodo4_fin[0]*risultati_periodo4_fin[3]:.6f} s^2\n")



# ==========================================
# SEZIONE FIT LINEARE 
# ==========================================
#LETTURA X, Y, SIGMA_Y
lunghezza_fit   = [0.225, 0.265, 0.320, 0.400]
T2_fit   = [risultati_periodo1_fin[0]**2                        , risultati_periodo2_fin[0]**2                         , risultati_periodo3_fin[0]**2                         , risultati_periodo4_fin[0]**2]
sigma_y = [2*risultati_periodo1_fin[0]*risultati_periodo1_fin[3], 2*risultati_periodo2_fin[0]*risultati_periodo2_fin[3], 2*risultati_periodo3_fin[0]*risultati_periodo3_fin[3], 2*risultati_periodo4_fin[0]*risultati_periodo4_fin[3]]
m, c, sig_m, sig_c, cov_mc, rho, *_ = lt.linear_fit_complete(lunghezza_fit, T2_fit, sigma_y)

lt.plot_generic_fit(lunghezza_fit, T2_fit, sigma_y, m, c, sig_m, sig_c, cov_mc, xlabel='lunghezza (m)', ylabel='T^2 (s^2)', title='Fit lineare T^2 vs lunghezza')
plt.grid()
plt.savefig('fit_lineare_T2_lunghezza.png')
plt.show()

lt.print_fit_results(m, c, sig_m, sig_c, cov_mc, rho)

g = 4*np.pi**2/m

sigma_g = g * (sig_m/m)

print(f"Accelerazione di gravità: g = {g:.4f} m/s^2")
print(f"Incertezza su g: sigma_g = {sigma_g:.4f} m/s^2")
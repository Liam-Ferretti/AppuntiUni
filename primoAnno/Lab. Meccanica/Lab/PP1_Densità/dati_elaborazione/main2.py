import numpy as np
import matplotlib.pyplot as plt
import lab_tools as lt
from scipy.stats import norm
import pandas as pd


RES_NONIO = 0.05 # mm
RES_PALMER = 0.01 # mm
RES_BILANCIA = 0.01 # g
RES_PROVETTA = 2 # ml
INCERTEZZA_PROVETTA = 0.5 # ml

# LETTURA E ELABORAZIONE DEI DATI NONIO

valori_altezza_nonio_30mis = lt.read_single_variable("altezzaNonio.txt")
risultati_altezza_nonio = lt.sample_statistic(valori_altezza_nonio_30mis)
print("Altezza misurata con il nonio:")
lt.print_sample_statistics(risultati_altezza_nonio, color="blue")

sigma_res_nonio = lt.resolution_uncertainty(RES_NONIO)

risultati_finali_altezza_nonio = lt.get_total_uncertainty(risultati_altezza_nonio[0], risultati_altezza_nonio[3], sigma_res_nonio)

lt.print_total_uncertainty(risultati_finali_altezza_nonio['mean'], risultati_finali_altezza_nonio['sigma_a'], risultati_finali_altezza_nonio['sigma_b'], risultati_finali_altezza_nonio['sigma_total'], "mm", color="blue")
lt.print_final_result("Altezza (nonio) usando i valori campionari" , risultati_finali_altezza_nonio['mean'], risultati_finali_altezza_nonio['sigma_total'], "mm", color="blue")

#LETTURA E ELABORAZIONE DEI DATI PALMER

valori_altezza_palmer_30mis_raw = lt.read_single_variable("altezzaPalmer.txt")
offset_palmer = lt.read_single_variable("offsetPalmer.txt")

offset_medio, sigma_offset_palmer, valori_altezza_palmer_30mis, info_offset = lt.calculate_offset(offset_palmer, valori_altezza_palmer_30mis_raw, False)
lt.print_offset_results(offset_medio, sigma_offset_palmer, info_offset, color="green")

risultati_campionari_altezza_palmer = lt.sample_statistic(valori_altezza_palmer_30mis)

print("\nAltezza misurata con il palmer (corretta):")
lt.print_sample_statistics(risultati_campionari_altezza_palmer, color="green")

sigma_res_palmer = lt.resolution_uncertainty(RES_PALMER)

sigma_b_palmer = lt.calculate_sigma_b([sigma_offset_palmer, sigma_res_palmer])

risultati_finali_altezza_palmer = lt.get_total_uncertainty(risultati_campionari_altezza_palmer[0], risultati_campionari_altezza_palmer[3], sigma_b_palmer)

lt.print_total_uncertainty(risultati_finali_altezza_palmer['mean'], risultati_finali_altezza_palmer['sigma_a'], risultati_finali_altezza_palmer['sigma_b'], risultati_finali_altezza_palmer['sigma_total'], "mm", color="green")
lt.print_final_result("Altezza (palmer) usando i valori campionari" , risultati_finali_altezza_palmer['mean'], risultati_finali_altezza_palmer['sigma_total'], "mm", color="green")

# SCELGO IL PALMER IN QUANTO LA SIGMA E' PIU' PICCOLA
# MISURO ED ELABORO I DATI SUL DIAMETRO (CON IL PALMER)

valori_diametro_palmer_30mis_raw = lt.read_single_variable("diametroPalmer.txt")
_, _, valori_diametro_palmer_30mis, _ = lt.calculate_offset(offset_palmer, valori_diametro_palmer_30mis_raw, False)

risultati_campionari_diametro_palmer = lt.sample_statistic(valori_diametro_palmer_30mis)

print("\nDiametro misurato con il palmer (corretto):")
lt.print_sample_statistics(risultati_campionari_diametro_palmer, color="yellow")

risultati_finali_diametro_palmer = lt.get_total_uncertainty(risultati_campionari_diametro_palmer[0], risultati_campionari_diametro_palmer[3], sigma_b_palmer)

lt.print_total_uncertainty(risultati_finali_diametro_palmer['mean'], risultati_finali_diametro_palmer['sigma_a'], risultati_finali_diametro_palmer['sigma_b'], risultati_finali_diametro_palmer['sigma_total'], "mm", color="yellow")
lt.print_final_result("Diametro (palmer) usando i valori campionari" , risultati_finali_diametro_palmer['mean'], risultati_finali_diametro_palmer['sigma_total'], "mm", color="yellow")


# MIISURO ED ELABORO I DATI SULLA MASSA

valori_massa_30mis = lt.read_single_variable("massa.txt")
risultati_campionari_massa = lt.sample_statistic(valori_massa_30mis)

print("\nMassa misurata con la bilancia:")
lt.print_sample_statistics(risultati_campionari_massa, color="red")

sigma_res_bilancia = lt.resolution_uncertainty(RES_BILANCIA)

risultati_finali_massa = lt.get_total_uncertainty(risultati_campionari_massa[0], risultati_campionari_massa[3], sigma_res_bilancia)

lt.print_total_uncertainty(risultati_finali_massa['mean'], risultati_finali_massa['sigma_a'], risultati_finali_massa['sigma_b'], risultati_finali_massa['sigma_total'], "g", color="red")
lt.print_final_result("Massa usando i valori campionari" , risultati_finali_massa['mean'], risultati_finali_massa['sigma_total'], "g", color="red")




#  CALCOLO LA DENSITA

volume_30mis = (1/4) * np.pi * valori_diametro_palmer_30mis**2 * valori_altezza_palmer_30mis / 1000

densità_30mis = valori_massa_30mis / volume_30mis

for i in range(len(densità_30mis)):
    print(densità_30mis[i])

# metodo 1: 30 densità, una per ogni misura + propagazione dell'incertezza usando la media delle singole propagazioni sulle sigma b e sommando in quadratura con la sigma a della densità campionaria
# calcolo media e sigma della densità campionaria

risultati_campionari_densità = lt.sample_statistic(densità_30mis)

sigma_tot_altezza  = np.sqrt(sigma_b_palmer**2 + risultati_campionari_altezza_palmer[2]**2)
sigma_tot_diametro = np.sqrt(sigma_b_palmer**2 + risultati_campionari_diametro_palmer[2]**2)
sigma_tot_massa    = np.sqrt(sigma_res_bilancia**2 + risultati_campionari_massa[2]**2)

prop_sigma_tot = densità_30mis * np.sqrt(
    (2 * sigma_tot_diametro / valori_diametro_palmer_30mis)**2 + 
    (sigma_tot_altezza / valori_altezza_palmer_30mis)**2 + 
    (sigma_tot_massa / valori_massa_30mis)**2
)
for i in range(len(densità_30mis)):
    print(f"Densità {i+1:<2}: {densità_30mis[i]:6.4f} g/cm³, sigma b = {prop_sigma_tot[i]:6.4f} g/cm³")

sigma_tot_densità_media = np.mean(prop_sigma_tot)

print("\nDensità campionaria:")
lt.print_sample_statistics(risultati_campionari_densità, color="bold")

risultati_finali_densità_campionaria = lt.get_total_uncertainty(risultati_campionari_densità[0], 0, sigma_tot_densità_media)
lt.print_total_uncertainty(risultati_finali_densità_campionaria['mean'], risultati_finali_densità_campionaria['sigma_a'], risultati_finali_densità_campionaria['sigma_b'], risultati_finali_densità_campionaria['sigma_total'], "g/cm³", color="bold")

lt.print_final_result("Densità usando i valori campionari" , risultati_finali_densità_campionaria['mean'], risultati_finali_densità_campionaria['sigma_total'], "g/cm³", color="bold")

sommapesi = np.sum(1/prop_sigma_tot**2)

numeratore = np.sum(densità_30mis / prop_sigma_tot**2)

rho_best = numeratore / sommapesi

sigma_tot_densità_a_media = 1/np.sqrt(sommapesi)

print(f"\n{'='*60}")
print(f"DENSITÀ MIGLIORE CAMPIONE A-0")
print(f"{'='*60}")
print(f"ρ̂ = {rho_best:.6f} g/cm³")
print(f"σ_ρ̂ = {sigma_tot_densità_a_media:.6f} g/cm³")
print(f"{'='*60}\n")

# metodo 2: calcolo densità usando i valori best

volume_da_best = (1/4) * np.pi * risultati_finali_diametro_palmer['mean']**2 * risultati_finali_altezza_palmer['mean'] / 1000

densità_da_best = risultati_finali_massa['mean'] / volume_da_best

sigma_densità_da_best = densità_da_best * np.sqrt(
    (2 * risultati_finali_diametro_palmer['sigma_total']/risultati_finali_diametro_palmer['mean'])**2 + 
    (risultati_finali_altezza_palmer['sigma_total']/risultati_finali_altezza_palmer['mean'])**2 + 
    (risultati_finali_massa['sigma_total']/risultati_finali_massa['mean'])**2
    )

lt.print_final_result("Densità usando i valori best" , densità_da_best, sigma_densità_da_best, "g/cm³", color="magenta")






# 3.3 MISURA DELLA DENSITÀ CAMPIONI A

altezza_campioni_a_raw = lt.read_single_variable("altezzaCampioni.txt")
diametro_campioni_a_raw = lt.read_single_variable("diametroCampioni.txt")
massa_campioni_a = lt.read_single_variable("massaCampioni.txt")

# Correzione delle misure dei campioni A usando l'offset del palmer
_, _, altezza_campioni_a, _ = lt.calculate_offset(offset_palmer, altezza_campioni_a_raw, False)
_, _, diametro_campioni_a, _ = lt.calculate_offset(offset_palmer, diametro_campioni_a_raw, False)\
        
volumi_campioni_a = (1/4) * np.pi * diametro_campioni_a**2 * altezza_campioni_a / 1000

sigma_totale_altezza  = np.sqrt(sigma_b_palmer**2 + risultati_campionari_altezza_palmer[2]**2)
sigma_totale_diametro = np.sqrt(sigma_b_palmer**2 + risultati_campionari_diametro_palmer[2]**2)
sigma_totale_massa    = np.sqrt(sigma_res_bilancia**2 + risultati_campionari_massa[2]**2)

print("\nSigma totale altezza: ", sigma_totale_altezza)
print("Sigma totale diametro: ", sigma_totale_diametro)
print("Sigma totale massa: ", sigma_totale_massa)

densità_campioni_a = massa_campioni_a / volumi_campioni_a

"""print("\nMisure dei campioni A (corrette):")
for i in range(len(altezza_campioni_a)):
    print(f"Campione A{i+1:<2}: "
          f"Altezza = {altezza_campioni_a[i]:6.4f} mm,  "
          f"Diametro = {diametro_campioni_a[i]:7.4f} mm,  "
          f"Massa = {massa_campioni_a[i]:6.4f} g,  "
          f"Densità = {densità_campioni_a[i]:6.4f} g/cm³")"""

risultati_campionari_campioni_a_densità = lt.sample_statistic(densità_campioni_a)

""" prop_sigma_a_densità_campioni = risultati_campionari_campioni_a_densità[0] * np.sqrt(
    (2 * sigma_b_palmer / diametro_campioni_a)**2 + 
    (sigma_b_palmer / altezza_campioni_a)**2 + 
    (sigma_res_bilancia / massa_campioni_a)**2
)

prop_sigma_b_densità_campioni = risultati_campionari_campioni_a_densità[0] * np.sqrt(
    (2 * sigma_b_palmer / diametro_campioni_a)**2 + 
    (sigma_b_palmer / altezza_campioni_a)**2 + 
    (sigma_res_bilancia / massa_campioni_a)**2
)

sigma_b_densità_a_media = np.mean(prop_sigma_b_densità_campioni) """

prop_sigma_tot_densità_campioni = densità_campioni_a * np.sqrt(
    (2 * sigma_totale_diametro / diametro_campioni_a)**2 + 
    (sigma_totale_altezza / altezza_campioni_a)**2 + 
    (sigma_totale_massa / massa_campioni_a)**2
)

for i in range(len(densità_campioni_a)):
    print(f"Campione A{i+1:<2}: Altezza = {altezza_campioni_a[i]:6.4f} mm, sigma altezza = {sigma_totale_altezza:6.4f} mm, Diametro = {diametro_campioni_a[i]:7.4f} mm, sigma diametro = {sigma_totale_diametro:6.4f} mm, Massa = {massa_campioni_a[i]:6.4f} g, sigma massa = {sigma_totale_massa:6.4f} g, Densità = {densità_campioni_a[i]:6.4f} g/cm³, Sigma totale = {prop_sigma_tot_densità_campioni[i]:6.4f} g/cm³")

sommapesi = np.sum(1/prop_sigma_tot_densità_campioni**2)

numeratore = np.sum(densità_campioni_a / prop_sigma_tot_densità_campioni**2)

rho_best = numeratore / sommapesi

sigma_tot_densità_a_media = 1/np.sqrt(sommapesi)

print(f"\n{'='*60}")
print(f"DENSITÀ MIGLIORE CAMPIONI A")
print(f"{'='*60}")
print(f"ρ̂ = {rho_best:.6f} g/cm³")
print(f"σ_ρ̂ = {sigma_tot_densità_a_media:.6f} g/cm³")
print(f"{'='*60}\n")

print("\nSigma totale media per i campioni A: ", sigma_tot_densità_a_media)
print(np.mean(prop_sigma_tot_densità_campioni))

print("\nDensità campionaria 20 cilindri:")
lt.print_sample_statistics(risultati_campionari_campioni_a_densità, color="bold")

risultati_finali_densità_a_campionaria = lt.get_total_uncertainty(risultati_campionari_campioni_a_densità[0], 0, sigma_tot_densità_a_media)
lt.print_total_uncertainty(risultati_finali_densità_a_campionaria['mean'], risultati_finali_densità_a_campionaria['sigma_a'], risultati_finali_densità_a_campionaria['sigma_b'], risultati_finali_densità_a_campionaria['sigma_total'], "g/cm³", color="bold")

lt.print_final_result("Densità 20 cilindri" , risultati_finali_densità_a_campionaria['mean'], risultati_finali_densità_a_campionaria['sigma_total'], "g/cm³", color="bold")

plt.figure(figsize=(8, 5))
 
# media e sigma dai dati
media = rho_best
sigma = risultati_campionari_campioni_a_densità[2]

bin_width = sigma

# numero di bin basato sulla larghezza = sigma
k = int(round((densità_campioni_a.max() - densità_campioni_a.min()) / bin_width, 0)) + 4

plt.xlim(media - 3*sigma, media + 3*sigma)
plt.hist(densità_campioni_a, bins=k, color='C0', label='Istogramma delle densità', edgecolor='black')
plt.xlabel("Densità [g/cm³]")
plt.ylabel("Frequenza")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("distribuzione_densita_campioni_no_gauss.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))

# numero di bin basato sulla larghezza = sigma
k = int(round((densità_campioni_a.max() - densità_campioni_a.min()) / bin_width, 0)) + 4

# gaussiana normalizzata (PDF)
xline = np.linspace(media - 3*sigma, media + 3*sigma, 1000)
yline = (1. / np.sqrt(2. * np.pi * sigma**2)) * np.exp(-(xline - media)**2 / (2 * sigma**2))

plt.hist(densità_campioni_a, bins=k, color='C0', label='Istogramma delle densità', edgecolor='black', density=True)
plt.plot(xline, yline, color='C1', label=f'Gaussiana (μ={media:.4f}, σ={sigma:.4f})')
plt.axvline(media, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Media: {media:.4f} g/cm³')
plt.xlabel("Densità [g/cm³]")
plt.ylabel("Frequenza normalizzata")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("distribuzione_densita_campioni.png", dpi=300)
plt.show()
# from scipy.stats import shapiro

# stat, p_value = shapiro(densità_campioni_a)
# print(f"Shapiro-Wilk test: p-value = {p_value:.4f}")
# if p_value > 0.05:
#     print("✓ Distribuzione compatibile con gaussiana (p > 0.05)")
# else:
#     print("✗ Distribuzione non significativamente gaussiana (p < 0.05)")
#
# from scipy.stats import shapiro
#RISULTATOOOOOOOO TODO
#Shapiro-Wilk test: p-value = 0.6084
#✓ Distribuzione compatibile con gaussiana (p > 0.05)




# 3.4 MISURA DELLA DENSITÀ CAMPIONE B

altezza_b = 88.5
larghezza_b = 20.000
lunghezza_b = 20.000
massa_b = 96.38

sigma_altezza_b_b = lt.resolution_uncertainty(RES_NONIO)
sigma_altezza_b_a = risultati_altezza_nonio[2]
sigma_altezza_b_total = np.sqrt(sigma_altezza_b_a**2 + sigma_altezza_b_b**2)

sigma_larghezza_b_b = lt.resolution_uncertainty(RES_NONIO)
sigma_larghezza_b_a = risultati_altezza_nonio[2]
sigma_larghezza_b_total = np.sqrt(sigma_larghezza_b_a**2 + sigma_larghezza_b_b**2)

sigma_lunghezza_b_b = lt.resolution_uncertainty(RES_NONIO)
sigma_lunghezza_b_a = risultati_altezza_nonio[2]
sigma_lunghezza_b_total = np.sqrt(sigma_lunghezza_b_a**2 + sigma_lunghezza_b_b**2)

sigma_massa_b_b = lt.resolution_uncertainty(RES_BILANCIA)
sigma_massa_b_a = risultati_campionari_massa[2]
sigma_massa_b_total = np.sqrt(sigma_massa_b_a**2 + sigma_massa_b_b**2)

volume_b = altezza_b * larghezza_b * lunghezza_b / 1000
densità_b = massa_b / volume_b

sigma_densità_b_total = densità_b * np.sqrt(
    (sigma_altezza_b_total / altezza_b)**2 +
    (sigma_larghezza_b_total / larghezza_b)**2 +
    (sigma_lunghezza_b_total / lunghezza_b)**2 +
    (sigma_massa_b_total / massa_b)**2
)
print("\n")
lt.print_final_result("Densità campione B" , densità_b, sigma_densità_b_total, "g/cm³", color="magenta")




# 3.5 MISURA DELLA DENSITÀ CAMPIONE C
altezza_c = 59
diametro_c = 9.95
massa_c = 38.79

sigma_altezza_c_b = lt.resolution_uncertainty(RES_NONIO)
sigma_altezza_c_a = risultati_altezza_nonio[2]
sigma_altezza_c_total = np.sqrt(sigma_altezza_c_a**2 + sigma_altezza_c_b**2)

sigma_diametro_c_b = lt.resolution_uncertainty(RES_NONIO)
sigma_diametro_c_a = risultati_altezza_nonio[2]
sigma_diametro_c_total = np.sqrt(sigma_diametro_c_a**2 + sigma_diametro_c_b**2)

sigma_massa_c_b = lt.resolution_uncertainty(RES_BILANCIA)
sigma_massa_c_a = risultati_campionari_massa[2]
sigma_massa_c_total = np.sqrt(sigma_massa_c_a**2 + sigma_massa_c_b**2)

volume_c = (1/4) * np.pi * diametro_c**2 * altezza_c / 1000
densità_c = massa_c / volume_c

sigma_densità_c_total = densità_c * np.sqrt(
    (sigma_altezza_c_total / altezza_c)**2 +
    (2 * sigma_diametro_c_total / diametro_c)**2 +
    (sigma_massa_c_total / massa_c)**2
)
print("\n")
lt.print_final_result("Densità campione C" , densità_c, sigma_densità_c_total, "g/cm³", color="magenta")




# 3.6 MISURAZIONE DELLE DENSITÀ DEI CAMPIONI A CON METODO GRAFICO

massa_integrale = lt.read_single_variable("massaPerGruppo.txt")
volume_integrale = lt.read_single_variable("volumePerGruppo.txt")

sigma_massa_b = lt.resolution_uncertainty(RES_BILANCIA)
sigma_massa_a = risultati_campionari_massa[2]
sigma_massa_total = np.sqrt(sigma_massa_a**2 + sigma_massa_b**2)

sigma_volume_a = 0 # non saprei come stimarla
sigma_volume_b = INCERTEZZA_PROVETTA

sigma_volume_total = np.sqrt(sigma_volume_a**2 + sigma_volume_b**2)

print("\nSigma massa totale: ", sigma_massa_total)
print("Sigma volume totale: ", sigma_volume_total)

# Fit NON pesato
risultati_fit = lt.linear_fit_complete(massa_integrale, volume_integrale)

m, c, sigma_m, sigma_c, cov_mc, rho, chi2, chi2_rid, *_ = risultati_fit

lt.print_fit_results(risultati_fit)

lt.plot_generic_fit(massa_integrale, volume_integrale, sigma_volume_total, 
                    m, c, sigma_m, sigma_c, cov_mc=cov_mc,
                    xlabel='Massa [g]', ylabel='Volume [cm³]', title='Fit lineare volume in funzione della massa', plot=True, save=True)

from scipy.stats import pearsonr

pearson_r, p_value = pearsonr(massa_integrale, volume_integrale)
print(f"Pearson r: {pearson_r:.4f}")

rho = 1 / m
sigma_rho = sigma_m / m**2

print(f"ρ = {rho:.4f} ± {sigma_rho:.4f} g/cm³")


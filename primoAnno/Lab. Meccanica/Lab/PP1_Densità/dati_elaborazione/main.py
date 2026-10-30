import numpy as np
import matplotlib.pyplot as plt
import lab_tools as lt

# =============================================================================
# CONFIGURAZIONE STRUMENTI
# Scheda §2.1 — risoluzione e incertezze nominali
# =============================================================================

# Calibro ventesimale (nonio): risoluzione 0.05 mm → sigma = 0.05/√12
# Palmer (micrometro):         risoluzione 0.01 mm → sigma = 0.01/√12  (conservativo)
# Bilancia:                    risoluzione 0.01 g  → sigma = 0.01/√12

RES_NONIO    = 0.05   # mm
RES_PALMER   = 0.01   # mm
RES_BILANCIA = 0.01   # g

# =============================================================================
# §3.1 — VERIFICA DEGLI STRUMENTI (offset)
# Il nonio ha zero ripetibile → sigma_offset già inclusa in sigma_res (nota 3)
# Il Palmer ha offset variabile → va misurato e corretto separatamente
# =============================================================================

offset_data = lt.read_single_variable("offsetPalmer.txt")

# L'offset viene calcolato qui e riusato in tutte le sezioni successive
# is_repeatable=False → distribuzione uniforme sull'intervallo osservato
offset_palmer, sigma_offset_palmer, _, info_offset = lt.calculate_offset(
    offset_data, offset_data, is_repeatable=False
)
# Nota: passiamo offset_data come "data" solo per ottenere sigma_offset;
# la correzione vera verrà applicata sezione per sezione sui dati reali.

# =============================================================================
# §3.2 — MISURA DENSITÀ DI UN SINGOLO CAMPIONE (30 misure ripetute)
# Obiettivo: valutazione incertezza Tipo A e Tipo B su altezza, diametro, massa
# Scelta dello strumento più adatto (confronto nonio vs Palmer)
# =============================================================================

print(lt.color_text("\n§3.2 — MISURA SINGOLO CAMPIONE (30 MISURE)", "bold"))

# --- §3.2.2  Confronto nonio vs Palmer sull'altezza ---

print(lt.color_text("\nNONIO", "blue"))
data_nonio = lt.read_single_variable("altezzaNonio.txt")
stats_n    = lt.sample_statistic(data_nonio)
lt.print_sample_statistics(stats_n, color="blue")

# Nonio: offset nullo ripetibile → sigma_b = solo sigma_res (nota 3 scheda)
sig_b_nonio = lt.calculate_sigma_b([lt.resolution_uncertainty(RES_NONIO)])
res_nonio   = lt.get_total_uncertainty(stats_n[0], stats_n[3], sig_b_nonio)
lt.print_total_uncertainty(res_nonio['mean'], res_nonio['sigma_a'],
                           res_nonio['sigma_b'], res_nonio['sigma_total'],
                           unit="mm", color="blue")

print(lt.color_text("\nPALMER", "red"))
raw_altezza_palmer = lt.read_single_variable("altezzaPalmer.txt")

# Correzione offset Palmer sull'altezza
offset, sigma_offset_palmer, clean_altezza_palmer, info_offset = lt.calculate_offset(
    offset_data, raw_altezza_palmer, is_repeatable=False
)
lt.print_offset_results(offset, sigma_offset_palmer, info_offset,
                        unit="mm", color="magenta")

stats_p   = lt.sample_statistic(clean_altezza_palmer)
lt.print_sample_statistics(stats_p, color="red")

# Palmer: sigma_b = sigma_res + sigma_offset in quadratura
sig_b_palmer = lt.calculate_sigma_b([lt.resolution_uncertainty(RES_PALMER),
                                     sigma_offset_palmer])
res_palmer   = lt.get_total_uncertainty(stats_p[0], stats_p[3], sig_b_palmer)
lt.print_total_uncertainty(res_palmer['mean'], res_palmer['sigma_a'],
                           res_palmer['sigma_b'], res_palmer['sigma_total'],
                           unit="mm", color="red")

# Confronto grafico delle distribuzioni dei due strumenti
mu1, s1 = res_nonio['mean'],  res_nonio['sigma_total']
mu2, s2 = res_palmer['mean'], res_palmer['sigma_total']
x_min = min(mu1, mu2) - 4 * max(s1, s2)
x_max = max(mu1, mu2) + 4 * max(s1, s2)

plt.figure(figsize=(10, 6))
lt.plot_normal(mean=mu1, sigma=s1, label="Nonio",  color="blue", unit="mm")
lt.plot_normal(mean=mu2, sigma=s2, label="Palmer", color="red",  unit="mm")
plt.xlim(x_min, x_max)
plt.title("§3.2 — Confronto Nonio vs Palmer (altezza)")
plt.xlabel("Altezza [mm]")
plt.ylabel("Densità di probabilità [mm⁻¹]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("confronto_distribuzioni.png", dpi=300)
plt.show()

# --- §3.2.3  Misura 30× diametro con Palmer (strumento scelto) e massa  ---
# Il Palmer ha sigma_tot minore → strumento più preciso per la densità

print(lt.color_text("\nDIAMETRO (Palmer, 30 misure)", "yellow"))
raw_diametro = lt.read_single_variable("diametroPalmer.txt")
# Nota: il diametro non è misurato attraverso lo zero → l'offset va comunque corretto
_, _, clean_diametro, _ = lt.calculate_offset(
    offset_data, raw_diametro, is_repeatable=False
)
stats_diametro = lt.sample_statistic(clean_diametro)
lt.print_sample_statistics(stats_diametro, color="yellow")

sig_b_diametro = lt.calculate_sigma_b([lt.resolution_uncertainty(RES_PALMER),
                                       sigma_offset_palmer])
res_diametro   = lt.get_total_uncertainty(stats_diametro[0], stats_diametro[3],
                                          sig_b_diametro)
lt.print_total_uncertainty(res_diametro['mean'], res_diametro['sigma_a'],
                           res_diametro['sigma_b'], res_diametro['sigma_total'],
                           unit="mm", color="yellow")

print(lt.color_text("\nMASSA (bilancia, 30 misure)", "green"))
massa = lt.read_single_variable("massa.txt")
stats_massa = lt.sample_statistic(massa)
lt.print_sample_statistics(stats_massa, color="green")

# Bilancia: nessun offset osservato → sigma_b = solo sigma_res
sig_b_massa = lt.calculate_sigma_b([lt.resolution_uncertainty(RES_BILANCIA)])
res_massa   = lt.get_total_uncertainty(stats_massa[0], stats_massa[3], sig_b_massa)
lt.print_total_uncertainty(res_massa['mean'], res_massa['sigma_a'],
                           res_massa['sigma_b'], res_massa['sigma_total'],
                           unit="g", color="green")

# --- §3.2.4–5  Calcolo densità dal singolo campione (30 terne h, d, m) ---
# Una densità per terna → Tipo A dalla dispersione delle densità
# Tipo B = sigma_rho propagata strumentalmente (media delle singole sigma_rho_i)

print(lt.color_text("\n§3.2.4–5 — DENSITÀ SINGOLO CAMPIONE (30 misure)", "green"))

densities_30, sigma_rho_30 = lt.calculate_density(
    clean_altezza_palmer, clean_diametro, massa,
    sigma_height=res_palmer['sigma_total'],
    sigma_diameter=res_diametro['sigma_total'],
    sigma_mass=res_massa['sigma_total']
)

stats_densita = lt.sample_statistic(densities_30)
lt.print_sample_statistics(stats_densita, color="green")

sig_b_densita = lt.calculate_sigma_b([np.mean(sigma_rho_30)])
res_densita   = lt.get_total_uncertainty(stats_densita[0], stats_densita[3],
                                         sig_b_densita)
lt.print_total_uncertainty(res_densita['mean'], res_densita['sigma_a'],
                           res_densita['sigma_b'], res_densita['sigma_total'],
                           unit="g/cm³", color="green")

# =============================================================================
# §3.3 — MISURA DENSITÀ DEI CAMPIONI A (misura singola per campione)
# Obiettivo: trasferire l'incertezza da misure ripetute a misura singola
# Tipo A = std dello strumento dalle 30 misure (senza dividere per √N)
# =============================================================================

print(lt.color_text("\n§3.3 — DENSITÀ CAMPIONI A (misura singola per campione)", "cyan"))

diametro_campioni = lt.read_single_variable("diametroCampioni.txt")
altezza_campioni  = lt.read_single_variable("altezzaCampioni.txt")
massa_campioni    = lt.read_single_variable("massaCampioni.txt")

# Correzione offset su ogni campione
_, sigma_offset_palmer, clean_diam_camp, _ = lt.calculate_offset(
    offset_data, diametro_campioni, is_repeatable=False
)
_, _,                   clean_alt_camp,  _ = lt.calculate_offset(
    offset_data, altezza_campioni,  is_repeatable=False
)

# Per misura singola: Tipo A = std dello strumento (dalle 30 misure ripetute)
# Non si divide per √N perché N=1 per ogni campione
sigma_a_altezza_camp = stats_p[2]   # std delle 30 misure di altezza Palmer
sigma_a_diametro_camp = stats_diametro[2]   # std delle 30 misure di diametro Palmer
sigma_a_massa_camp = stats_massa[2]

print("\n Sigma Tipo A campioni (alteza): " + f"{sigma_a_altezza_camp:.4f} mm")
print(" Sigma Tipo A campioni (diametro): " + f"{sigma_a_diametro_camp:.4f} mm")
print(" Sigma Tipo A massa: " + f"{sigma_a_massa_camp:.4f} g")

sig_b_diam_camp = lt.calculate_sigma_b([lt.resolution_uncertainty(RES_PALMER),
                                        sigma_offset_palmer])
sig_b_alt_camp  = lt.calculate_sigma_b([lt.resolution_uncertainty(RES_PALMER),
                                        sigma_offset_palmer])
sig_b_massa_camp  = lt.calculate_sigma_b(lt.resolution_uncertainty(RES_BILANCIA))

sigma_tot_alt_camp  = np.sqrt(sigma_a_altezza_camp**2 + sig_b_alt_camp**2)
sigma_tot_diam_camp = np.sqrt(sigma_a_diametro_camp**2 + sig_b_diam_camp**2)
sigma_tot_massa_camp  = np.sqrt(sigma_a_massa_camp**2 + sig_b_massa_camp**2)

densities_camp, sigma_rho_camp = lt.calculate_density(
    clean_alt_camp, clean_diam_camp, massa_campioni,
    sigma_height=sigma_tot_alt_camp,
    sigma_diameter=sigma_tot_diam_camp,
    sigma_mass=sigma_tot_massa_camp
)

print("\nDensità campioni A:")
for i, (rho, sigma) in enumerate(zip(densities_camp, sigma_rho_camp)):
    print(f"  Campione {i+1:2d}: {rho:.4f} ± {sigma:.4f} g/cm³")

# §3.3.4 — Istogramma densità campioni A
k = int(np.ceil(1 + np.log2(len(densities_camp))))   # regola di Sturges
plt.figure()
plt.hist(densities_camp, bins=k, alpha=0.7, color="cyan", edgecolor="black")
plt.title(f"§3.3 — Distribuzione densità campioni A ({k} bins, regola di Sturges)")
plt.xlabel("Densità [g/cm³]")
plt.ylabel("Frequenza")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("distribuzione_densita_campioni.png", dpi=300)
plt.show()

# §3.3.5 — Miglior stima densità campioni A
stats_densita_camp = lt.sample_statistic(densities_camp)
sig_b_densita_camp = np.mean(sigma_rho_camp)
res_densita_camp   = lt.get_total_uncertainty(stats_densita_camp[0],
                                               stats_densita_camp[3],
                                               sig_b_densita_camp)

lt.print_total_uncertainty(res_densita_camp['mean'], res_densita_camp['sigma_a'],
                           res_densita_camp['sigma_b'], res_densita_camp['sigma_total'],
                           unit="g/cm³", color="cyan")

print(res_densita_camp)

# =============================================================================
# §3.6 — METODO GRAFICO: fit lineare V = (1/ρ)·M
# Obiettivo: stimare ρ dalla pendenza e confrontare con metodi precedenti
# V = M/ρ  →  pendenza m = 1/ρ  →  ρ = 1/m
# =============================================================================

print(lt.color_text("\n§3.6 — METODO GRAFICO: FIT LINEARE V(M)", "magenta"))

# Volume di ogni campione (in cm³, da mm³ → ÷1000)
volume_campioni = np.pi * clean_diam_camp**2 / 4 * clean_alt_camp / 1000

# Propagazione incertezza sul volume: (σ_V/V)² = (2σ_d/d)² + (σ_h/h)²
sigma_volume = volume_campioni * np.sqrt(
    (2 * sigma_tot_diam_camp / clean_diam_camp)**2 +
    (sigma_tot_alt_camp      / clean_alt_camp)**2
)

# --- FIT 1: pesato con sigma teorica propagata ---
print(lt.color_text("\nFIT 1 — Pesato con sigma teorica", "yellow"))
fit_teorico = lt.linear_fit_complete(massa_campioni, volume_campioni, sigma_volume)
lt.print_fit_results(fit_teorico)
m1, c1, sm1, sc1, cov1, *_ = fit_teorico
rho1       = 1 / m1
sigma_rho1 = sm1 / m1**2

# --- FIT 2: metodo dei residui (sigma stimata empiricamente) ---
# Utile perché il chi²_rid > 1 del fit teorico indica sigma teorica sottostimata
print(lt.color_text("\nFIT 2 — Metodo dei residui (sigma empirica)", "green"))
fit_residui, sigma_emp, residui = lt.linear_fit_residuals(massa_campioni, volume_campioni)
lt.print_fit_results(fit_residui)
m2, c2, sm2, sc2, cov2, *_ = fit_residui
rho2       = 1 / m2
sigma_rho2 = sm2 / m2**2
print(f"  sigma empirica dai residui: {sigma_emp:.6f} cm³")

print(lt.color_text(f"\nDensità fit minimi quadrati:  ρ = {rho1:.4f} ± {sigma_rho1:.4f} g/cm³", "yellow"))
print(lt.color_text(f"Densità fit residui:  ρ = {rho2:.4f} ± {sigma_rho2:.4f} g/cm³", "green"))

# --- Grafico confronto fit ---
x_range    = np.linspace(min(massa_campioni), max(massa_campioni), 200)
sigma_plot = np.full_like(volume_campioni, sigma_emp)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, (m, c, sm, sc, cov, sigma_y, label, color) in zip(axes, [
    (m1, c1, sm1, sc1, cov1, sigma_volume, "Fit minimi quadrati", "royalblue"),
    (m2, c2, sm2, sc2, cov2, sigma_plot,   "Fit residui", "tomato"),
]):
    y_fit      = m * x_range + c
    sigma_band = np.sqrt((x_range * sm)**2 + sc**2 + 2 * x_range * cov)

    ax.errorbar(massa_campioni, volume_campioni, yerr=sigma_y,
                fmt='o', color='black', capsize=4, label='Dati', zorder=3)
    ax.plot(x_range, y_fit, color=color,
            label=f'{label}\nm={m:.4f}, c={c:.4f}')
    ax.fill_between(x_range, y_fit - sigma_band, y_fit + sigma_band,
                    alpha=0.2, color=color, label='Banda ±σ')
    ax.set_title(label)
    ax.set_xlabel("Massa [g]")
    ax.set_ylabel("Volume [cm³]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("§3.6 — Confronto fit minimi quadrati vs metodo dei residui", fontsize=13)
plt.tight_layout()
plt.savefig("confronto_fit.png", dpi=300)
plt.show()

# --- Grafico residui ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for ax, (m, c, sigma_y, label, color) in zip(axes, [
    (m1, c1, sigma_volume, "Fit teorico", "royalblue"),
    (m2, c2, sigma_plot,   "Fit residui", "tomato"),
]):
    res = volume_campioni - (m * massa_campioni + c)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.errorbar(massa_campioni, res, yerr=sigma_y,
                fmt='o', color=color, capsize=4)
    ax.set_title(f"Residui — {label}")
    ax.set_xlabel("Massa [g]")
    ax.set_ylabel("Residuo [cm³]")
    ax.grid(True, alpha=0.3)

plt.suptitle("§3.6 — Confronto residui", fontsize=13)
plt.tight_layout()
plt.savefig("confronto_residui.png", dpi=300)
plt.show()

# =============================================================================
# RIEPILOGO FINALE
# =============================================================================

print("\n" + "=" * 60)
print(lt.color_text("RIEPILOGO RISULTATI FINALI", "bold"))
print("=" * 60)
lt.print_final_result("Altezza (Nonio)",  res_nonio['mean'],        res_nonio['sigma_total'],        "mm",    color="blue")
lt.print_final_result("Altezza (Palmer)", res_palmer['mean'],       res_palmer['sigma_total'],       "mm",    color="red")
lt.print_final_result("Diametro",         res_diametro['mean'],     res_diametro['sigma_total'],     "mm",    color="yellow")
lt.print_final_result("Massa",            res_massa['mean'],        res_massa['sigma_total'],        "g",     color="green")
lt.print_final_result("Densità campionaria singolo cilindro §3.2",     res_densita['mean'],      res_densita['sigma_total'],      "g/cm³", color="green")
lt.print_final_result("Densità dei 20 cilindri §3.3",     res_densita_camp['mean'], res_densita_camp['sigma_total'], "g/cm³", color="cyan")
lt.print_final_result("Densità §3.6 (fit minimi quadrati)", rho1, sigma_rho1, "g/cm³", color="yellow")
lt.print_final_result("Densità §3.6 (fit residui)", rho2, sigma_rho2, "g/cm³", color="magenta")



# metodo 2: densità con best h, d, m

best_dens = (res_massa["mean"] / ((1/4) * np.pi * res_diametro['mean']**2 * res_palmer["mean"])) * 1000

sigma_best_dens = best_dens * np.sqrt((2 * res_diametro["sigma_total"]/res_diametro["mean"])**2 + (res_palmer["sigma_total"]/res_palmer["mean"])**2 + (res_massa["sigma_total"]/res_massa["mean"])**2)

lt.print_final_result("Densità usando i val best" , best_dens, sigma_best_dens, "g/cm³", color="purple")
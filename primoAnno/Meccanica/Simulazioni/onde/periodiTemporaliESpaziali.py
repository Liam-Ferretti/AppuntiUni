import matplotlib.pyplot as plt
import numpy as np

# Parametri differenziati per rompere la simmetria
A = 4
k = 0.8       # Spazio più "dilatato" (onda lunga)
omega = 1.5   # Tempo più "veloce" (oscillazione rapida)

# Calcolo analitico
lambda_spaziale = 2 * np.pi / k     # ~ 7.85
T_temporale = 2 * np.pi / omega     # ~ 4.19

# Limiti degli assi differenziati per adattarsi ai rispettivi periodi
x_max = 12
t_max = 8

fig = plt.figure(figsize=(18, 6))

# -------------------------------------------------------------------------
# 1. GRAFICO SPAZIALE (y vs x) a t = 0
# -------------------------------------------------------------------------
ax1 = fig.add_subplot(1, 3, 1)
x = np.linspace(0, x_max, 1000)
y_spazio = A * np.cos(k * x)

ax1.plot(x, y_spazio, color='#1f77b4', linewidth=2.5)
ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=lambda_spaziale, color='red', linestyle='--', linewidth=1.5)

# Freccia e testo per Lambda con formula LaTeX
ax1.annotate('', xy=(0, A * 0.4), xytext=(lambda_spaziale, A * 0.4),
             arrowprops=dict(arrowstyle='<->', color='red', lw=2))

# Concateniamo il testo con la formula LaTeX per evitare bug di newline
testo_lambda = "Lunghezza d'onda\n" + r"$\lambda = \frac{2\pi}{k} = " + f"{lambda_spaziale:.2f}$"
ax1.text(lambda_spaziale / 2, A * 0.5, testo_lambda, 
         color='red', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_title("Profilo Spaziale ($t = 0$)\n$y(x, 0) = A \\cos(kx)$", fontsize=12, pad=10)
ax1.set_xlabel('Spazio (x)')
ax1.set_ylabel('Ampiezza (y)')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_xlim(0, x_max)
ax1.set_ylim(-A * 1.4, A * 1.4)

# -------------------------------------------------------------------------
# 2. GRAFICO TEMPORALE (y vs t) a x = 0
# -------------------------------------------------------------------------
ax2 = fig.add_subplot(1, 3, 2)
t = np.linspace(0, t_max, 1000)
y_tempo = A * np.cos(-omega * t)

ax2.plot(t, y_tempo, color='#2ca02c', linewidth=2.5)
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=T_temporale, color='red', linestyle='--', linewidth=1.5)

# Freccia e testo per il Periodo T con formula LaTeX
ax2.annotate('', xy=(0, -A * 0.4), xytext=(T_temporale, -A * 0.4),
             arrowprops=dict(arrowstyle='<->', color='red', lw=2))

testo_periodo = "Periodo Temporale\n" + r"$T = \frac{2\pi}{\omega} = " + f"{T_temporale:.2f}$"
ax2.text(T_temporale / 2, -A * 0.5, testo_periodo, 
         color='red', ha='center', va='top', fontsize=10, fontweight='bold')

ax2.set_title("Evoluzione Temporale ($x = 0$)\n$y(0, t) = A \\cos(-\\omega t)$", fontsize=12, pad=10)
ax2.set_xlabel('Tempo (t)')
ax2.set_ylabel('Ampiezza (y)')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.set_xlim(0, t_max)
ax2.set_ylim(-A * 1.4, A * 1.4)

# -------------------------------------------------------------------------
# 3. GRAFICO SPAZIO-TEMPORALE 3D (Griglia Rettangolare)
# -------------------------------------------------------------------------
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

X, T = np.meshgrid(np.linspace(0, x_max, 100), np.linspace(0, t_max, 100))
Y = A * np.cos(k * X - omega * T)

surf = ax3.plot_surface(X, T, Y, cmap='coolwarm', edgecolor='none', alpha=0.85)

ax3.set_title("Funzione d'Onda Completa\n$y(x, t) = A \\cos(kx - \\omega t)$", fontsize=12, pad=15)
ax3.set_xlabel('Spazio (x)', fontsize=9)
ax3.set_ylabel('Tempo (t)', fontsize=9)
ax3.set_zlabel('Ampiezza (y)', fontsize=9)

ax3.view_init(elev=30, azim=-130)
fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=12, pad=0.1)

plt.tight_layout()
plt.savefig("output/periodiTemporaliESpazialiOndePiane.jpg", dpi=300)
plt.show()
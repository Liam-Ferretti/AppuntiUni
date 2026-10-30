import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def calcola_moto(t, zeta, w0, x0, v0):
    """Calcola la posizione x(t) per qualsiasi valore di zeta."""
    if zeta < 1: # Sottosmorzato
        wd = w0 * np.sqrt(1 - zeta**2)
        A = x0
        B = (v0 + zeta * w0 * x0) / wd
        return np.exp(-zeta * w0 * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))
    
    elif zeta == 1: # Criticamente smorzato
        A = x0
        B = v0 + w0 * x0
        return (A + B * t) * np.exp(-w0 * t)
    
    else: # Sovrasmorzato (quello nell'immagine)
        r1 = w0 * (-zeta + np.sqrt(zeta**2 - 1))
        r2 = w0 * (-zeta - np.sqrt(zeta**2 - 1))
        C2 = (v0 - r1 * x0) / (r2 - r1)
        C1 = x0 - C2
        return C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)

# --- Configurazione Grafica ---
t = np.linspace(0, 10, 1000)
zeta_init, w0_init, x0_init, v0_init = 1.5, 1.0, 0.5, 3.0

fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.3) # Spazio per gli slider

x_vals = calcola_moto(t, zeta_init, w0_init, x0_init, v0_init)
line, = ax.plot(t, x_vals, lw=2, color='#e76f51')
ax.axhline(0, color='black', lw=1, ls='--')
ax.set_ylim(-5, 5)
ax.set_title("Simulatore Moto Smorzato")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Posizione x(t)")

# --- Creazione Slider ---
ax_color = 'lightgoldenrodyellow'
ax_zeta = plt.axes([0.2, 0.2, 0.6, 0.03], facecolor=ax_color)
ax_w0   = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor=ax_color)
ax_x0   = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=ax_color)
ax_v0   = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor=ax_color)

s_zeta = Slider(ax_zeta, 'Zeta (ζ)', 0.01, 3.0, valinit=zeta_init)
s_w0   = Slider(ax_w0, 'ω0', 0.1, 5.0, valinit=w0_init)
s_x0   = Slider(ax_x0, 'x0', -2.0, 2.0, valinit=x0_init)
s_v0   = Slider(ax_v0, 'v0', -5.0, 5.0, valinit=v0_init)

def update(val):
    zeta = s_zeta.val
    w0 = s_w0.val
    x0 = s_x0.val
    v0 = s_v0.val
    line.set_ydata(calcola_moto(t, zeta, w0, x0, v0))
    
    # Cambio colore in base al regime
    if zeta < 1: line.set_color('#264653')   # Blu (Sotto)
    elif zeta == 1: line.set_color('#2a9d8f') # Verde (Critico)
    else: line.set_color('#e76f51')           # Arancio (Sovra)
    
    fig.canvas.draw_idle()

s_zeta.on_changed(update)
s_w0.on_changed(update)
s_x0.on_changed(update)
s_v0.on_changed(update)

plt.show()
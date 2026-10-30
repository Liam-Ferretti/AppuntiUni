import cmath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc


def rappresenta_somma_vettori_completo(z1, z2):
    """Rappresenta la somma di due numeri complessi includendo tutti gli angoli

    dei tuoi appunti: alpha_1, alpha_2, alpha, phi, gamma_1 e gamma_2.
    """
    # Calcolo del vettore somma
    z_somma = z1 + z2

    # Estrazione moduli e argomenti (in gradi per la compatibilità con Arc)
    r1, theta1 = cmath.polar(z1)
    r2, theta2 = cmath.polar(z2)
    r_s, theta_s = cmath.polar(z_somma)

    deg1, deg2, deg_s = (
        np.degrees(theta1),
        np.degrees(theta2),
        np.degrees(theta_s),
    )

    fig, ax = plt.subplots(figsize=(9, 9))

    # --- 1. DISEGNO DEI VETTORI PRINCIPALI ---
    # Vettore Z1 (Blu)
    ax.quiver(
        0,
        0,
        z1.real,
        z1.imag,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#1f77b4",
        linewidth=2,
        zorder=5,
    )
    ax.text(
        z1.real * 0.7,
        z1.imag * 0.6,
        r"$\vec{A}_1$",
        color="#1f77b4",
        fontsize=14,
        fontweight="bold",
    )

    # Vettore Z2 (Verde)
    ax.quiver(
        0,
        0,
        z2.real,
        z2.imag,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#2ca02c",
        linewidth=2,
        zorder=5,
    )
    ax.text(
        z2.real * 0.5,
        z2.imag * 0.7,
        r"$\vec{A}_2$",
        color="#2ca02c",
        fontsize=14,
        fontweight="bold",
    )

    # Vettore Somma Z (Rosso)
    ax.quiver(
        0,
        0,
        z_somma.real,
        z_somma.imag,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#d62728",
        linewidth=2.5,
        zorder=6,
    )
    ax.text(
        z_somma.real * 0.6,
        z_somma.imag * 0.65,
        r"$\vec{A}$",
        color="#d62728",
        fontsize=15,
        fontweight="bold",
    )

    # --- 2. COSTRUZIONE PARALLELOGRAMMA ---
    ax.plot(
        [z1.real, z_somma.real],
        [z1.imag, z_somma.imag],
        color="#2ca02c",
        linestyle=":",
        linewidth=1.2,
    )
    ax.plot(
        [z2.real, z_somma.real],
        [z2.imag, z_somma.imag],
        color="#1f77b4",
        linestyle=":",
        linewidth=1.2,
    )

    # Raggio base di riferimento per scalare gli archi
    r_base = min(r1, r2)

    # --- 3. ANGOLI ASSOLUTI (RISPETTO RE) ---
    # Arco alpha_1
    r_a1 = r_base * 0.20
    ax.add_patch(
        Arc((0, 0), r_a1 * 2, r_a1 * 2, theta1=0, theta2=deg1, color="#1f77b4")
    )
    ax.text(
        r_a1 * 1.35 * np.cos(np.radians(deg1 / 2)),
        r_a1 * 1.35 * np.sin(np.radians(deg1 / 2)),
        r"$\alpha_1$",
        color="#1f77b4",
        fontsize=11,
    )

    # Arco alpha_2
    r_a2 = r_base * 0.30
    ax.add_patch(
        Arc((0, 0), r_a2 * 2, r_a2 * 2, theta1=0, theta2=deg2, color="#2ca02c")
    )
    ax.text(
        r_a2 * 1.25 * np.cos(np.radians(deg2 * 0.85)),
        r_a2 * 1.25 * np.sin(np.radians(deg2 * 0.85)),
        r"$\alpha_2$",
        color="#2ca02c",
        fontsize=11,
    )

    # Arco alpha (vettore somma - esterno)
    r_a = r_s * 0.55
    ax.add_patch(
        Arc(
            (0, 0),
            r_a * 2,
            r_a * 2,
            theta1=0,
            theta2=deg_s,
            color="#d62728",
            linestyle="--",
        )
    )
    ax.text(
        r_a * 1.12 * np.cos(np.radians(deg_s * 0.6)),
        r_a * 1.12 * np.sin(np.radians(deg_s * 0.6)),
        r"$\alpha$",
        color="#d62728",
        fontsize=13,
    )

    # --- 4. ANGOLI INTERNI ED INTERMEDI ---
    # Arco phi (sfasamento totale tra A1 e A2)
    r_ph = r_base * 0.45
    ax.add_patch(
        Arc(
            (0, 0),
            r_ph * 2,
            r_ph * 2,
            theta1=deg1,
            theta2=deg2,
            color="#e08b00",
            linewidth=1.5,
        )
    )
    meta_phi = deg1 + (deg2 - deg1) / 2
    ax.text(
        r_ph * 1.2 * np.cos(np.radians(meta_phi)),
        r_ph * 1.2 * np.sin(np.radians(meta_phi)),
        r"$\phi$",
        color="#e08b00",
        fontsize=13,
        fontweight="bold",
    )

    # Arco gamma_1 (tra A1 e Somma A)
    r_g1 = r_base * 0.60
    ax.add_patch(
        Arc(
            (0, 0),
            r_g1 * 2,
            r_g1 * 2,
            theta1=deg1,
            theta2=deg_s,
            color="#9467bd",
            linewidth=1.2,
        )
    )
    meta_g1 = deg1 + (deg_s - deg1) / 2
    ax.text(
        r_g1 * 1.15 * np.cos(np.radians(meta_g1)),
        r_g1 * 1.15 * np.sin(np.radians(meta_g1)),
        r"$\gamma_1$",
        color="#9467bd",
        fontsize=12,
    )

    # Arco gamma_2 (tra Somma A e A2)
    r_g2 = r_base * 0.72
    ax.add_patch(
        Arc(
            (0, 0),
            r_g2 * 2,
            r_g2 * 2,
            theta1=deg_s,
            theta2=deg2,
            color="#17becf",
            linewidth=1.2,
        )
    )
    meta_g2 = deg_s + (deg2 - deg_s) / 2
    ax.text(
        r_g2 * 1.12 * np.cos(np.radians(meta_g2)),
        r_g2 * 1.12 * np.sin(np.radians(meta_g2)),
        r"$\gamma_2$",
        color="#17becf",
        fontsize=12,
    )

    # --- 5. FINISH GRAFICO ---
    ax.set_xlabel(r"$Re(z)$", fontsize=12, loc="right")
    ax.set_ylabel(r"$Im(z)$", fontsize=12, loc="top")

    max_val = max(z_somma.real, z_somma.imag)
    ax.set_xlim(-max_val * 0.1, max_val * 1.25)
    ax.set_ylim(-max_val * 0.1, max_val * 1.25)

    ax.axhline(0, color="black", linewidth=1.2, zorder=2)
    ax.axvline(0, color="black", linewidth=1.2, zorder=2)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.savefig("output/sommaOnde.png", dpi=300, bbox_inches="tight")
    plt.show()


# Test con valori geometrici simili al disegno
vettore_blu = cmath.rect(7, np.radians(20))  # A1
vettore_verde = cmath.rect(4.5, np.radians(65))  # A2

rappresenta_somma_vettori_completo(vettore_blu, vettore_verde)
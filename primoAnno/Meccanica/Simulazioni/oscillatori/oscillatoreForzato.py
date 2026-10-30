import numpy as np
import matplotlib.pyplot as plt

F, m, omega0 = 1.0, 1.0, 2.5

omega = np.linspace(0, 5.0, 1000)

betas = [2.0, 0.8, 0.2, 0.04]

def A(omega, beta):
    return (F / m) / np.sqrt((omega0**2 - omega**2)**2 + (beta * omega / m)**2)

# Su un solo grafico:
for beta in betas:
    plt.plot(omega, A(omega, beta), label=rf'$\beta = {beta}$')

plt.title("Ampiezza di un Oscillatore Forzato")
plt.xlabel(r'$\omega$ (rad/s)')
plt.ylabel(r'$A(\omega)$')
plt.axvline(omega0, color='gray', lw=1, ls='--')
plt.legend()
plt.grid()
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.tight_layout()
plt.savefig("output/oscillatore_forzato.png", dpi=300)
plt.show()
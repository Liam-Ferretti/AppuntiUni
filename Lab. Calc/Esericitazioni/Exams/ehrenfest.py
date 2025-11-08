import numpy as np
import matplotlib.pyplot as plt

x, y = np.loadtxt('dati.dat', unpack=True, usecols=(0, 1))

plt.figure(figsize=(7,5))
plt.scatter(x, y, marker='D', alpha=0.5)  # 'D' is for diamond markers
plt.xlabel("Colonna 0")
plt.ylabel("Colonna 1")
plt.title("Scatter plot con markers a diamante - dati.dat")
plt.tight_layout()
plt.show()

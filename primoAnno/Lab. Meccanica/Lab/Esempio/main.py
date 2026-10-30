import numpy as np
import matplotlib.pyplot as plt

dati = np.loadtxt('dati.csv', delimiter=',', skiprows=1, usecols=(0, 1))

t, val = dati[:, 0], dati[:, 1]

plt.plot(t, val)
plt.xlabel('Tempo')
plt.ylabel('Valore')
plt.title('Dati sperimentali')
plt.show()
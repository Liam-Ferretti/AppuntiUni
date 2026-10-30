import matplotlib.pyplot as plt

dati = []
with open('dati.txt', 'r') as file:
    for riga in file:

        valore = riga.strip()
        if valore:
            dati.append(float(valore))

plt.figure(figsize=(8, 5))

plt.hist(dati, bins=34, color='#0504aa', alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.5)
plt.xlabel('Valore')
plt.ylabel('Frequenza')
plt.title('Istogramma')
plt.show()
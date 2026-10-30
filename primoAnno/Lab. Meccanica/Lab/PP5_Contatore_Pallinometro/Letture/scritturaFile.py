import re
import matplotlib.pyplot as plt
import numpy as np

def from_file_to_list(file_path):
    dati = []
    try:
        with open(file_path, 'r') as file:
            for line in file:

                elements = line.strip().split('\t')

                dati.append(elements)

    except FileNotFoundError:
        print("Il file non è stato trovato.")

    return dati


LOCALPATH = './'

CoolTermFile = f'LetturaTufoOrizzontale.txt'

DataFile = f'lettura7.txt'

# Chiamata alla funzione per leggere i dati
file_path = LOCALPATH + CoolTermFile

dati_list = from_file_to_list(file_path)

# Inizializza i contenitori
results = []
total_counter = 0
current_time = 0.0  # Usiamo float per sicurezza

for row in dati_list:
    line_content = " ".join(row)
    
    matches = re.findall(r't([0-9A-Fa-f]+)', line_content)

    for match in matches:
        # Converti hex in decimale e poi in secondi
        offset_micros = int(match, 16)
        bin_time = offset_micros * 1e-6 
        
        total_counter += 1
        results.append((current_time + bin_time, total_counter))

    current_time += 1.0

with open(DataFile, "w") as f:
    for timestamp, count in results:
        # Formato: Conteggio [tab] Tempo
        f.write(f"{count}\t{timestamp:.6f}\n")

print(f"Elaborazione completata. Salvati {len(results)} eventi in {DataFile}.")


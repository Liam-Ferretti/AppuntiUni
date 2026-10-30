import lab_tools as lt
import numpy as np

val = 6.10
sigma = 0.0032

val, sigma = lt.round_to_sig_figs(val, sigma, 2)

print(f"Valore arrotondato: {val} ± {sigma}")
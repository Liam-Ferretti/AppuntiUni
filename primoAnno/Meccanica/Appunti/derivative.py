from latex2sympy2 import latex2sympy

tex_code = r"\frac{d}{dx}(\sin(x))"

operazione = latex2sympy(tex_code)

# .doit() esegue effettivamente il calcolo della derivata
risultato = operazione.doit()

print(f"Operazione: {operazione}")
print(f"Risultato calcolato: {risultato}")
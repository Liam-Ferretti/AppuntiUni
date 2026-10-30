"""
lab_tools.py
============
Libreria di utility per il laboratorio di meccanica.

CONFIGURAZIONE GLOBALE
----------------------
RETURN_TUPLE : bool (default: True)
    Se True, get_total_uncertainty restituisce una tupla (mean, sigma_a, sigma_b, sigma_total).
    Se False, restituisce un dizionario {'mean': ..., 'sigma_a': ..., ...}.

Funzioni disponibili
--------------------
Lettura dati:
    read_single_variable(filename)      Legge un file con una sola colonna
    read_xy_data(filename)              Legge un file con 2 o 3 colonne (x, y, sigma_y)
    read_multiple_variables(filename)   Legge n colonne di lunghezza non uniforme

Incertezze:
    resolution_uncertainty(res)         Calcola sigma di risoluzione (res/sqrt(12))
    calculate_offset(...)               Corregge i dati sottraendo l'offset e stima sigma_offset
    sample_statistic(data)              Calcola media, varianza, std su misure ripetute
    calculate_sigma_b(sigma_b_list)     Combina i contributi Tipo B in quadratura
    get_total_uncertainty(...)          Combina incertezza Tipo A e Tipo B

Fit lineare:
    linear_fit_complete(x, y, sigma_y)  Fit pesato/non pesato y = m*x + c
    linear_fit_residuals(x, y)          Fit pesato con sigma stimata dai residui

Stampa e visualizzazione:
    print_sample_statistics(results)    Stampa le statistiche campionarie
    print_offset_results(...)           Stampa il riepilogo dell'offset e la correzione applicata
    print_total_uncertainty(...)        Stampa il riepilogo delle incertezze combinate
    print_final_result(...)             Stampa il risultato finale compatto
    print_fit_results(results)          Stampa i risultati del fit
    plot_generic_fit(...)               Grafico dati + fit + banda di incertezza
    plot_normal(...)                    Disegna una distribuzione normale sul grafico corrente

Workflow tipico
---------------
1. read_single_variable / read_xy_data / read_multiple_variables
2. resolution_uncertainty
3. calculate_offset           (se necessario)
4. print_offset_results       (se necessario)
5. sample_statistic
6. calculate_sigma_b          (se necessario)
7. get_total_uncertainty
8. print_sample_statistics / print_total_uncertainty / print_final_result
9. linear_fit_complete        (se necessario)
10. print_fit_results          (se necessario)
11. plot_generic_fit           (se necessario)

Dipendenze
----------
- numpy
- matplotlib

Autore: Liam Ferretti
Versione: 1.3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# =============================================================================
# CONFIGURAZIONE GLOBALE
# =============================================================================

RETURN_TUPLE = True  # Se True, get_total_uncertainty restituisce tupla; se False, dizionario

# =============================================================================
# COSTANTI E UTILITY
# =============================================================================

# Costanti per i colori nel terminale (ANSI escape codes)
COLORS = {
    "blue": "\033[94m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "magenta": "\033[95m",
    "bold": "\033[1m",
    "reset": "\033[0m"
}

def _apply_color(text, color_name):
    """
    Funzione interna per applicare i codici ANSI di colore al testo.
    
    Parametri
    ----------
    text : str
        Il testo da colorare.
    color_name : str
        Nome del colore ('blue', 'red', 'green', 'yellow', 'magenta', 'bold', 'reset').
    
    Restituisce
    -------
    colored_text : str
        Il testo con i codici ANSI applicati, o il testo originale se colore non trovato.
    
    Esempio
    -------
    >>> colored = _apply_color("Attenzione", "red")
    >>> print(colored)  # Stampa "Attenzione" in rosso
    """
    if color_name in COLORS:
        return f"{COLORS[color_name]}{text}{COLORS['reset']}"
    return text

def color_text(text, color_name):
    """
    Funzione pubblica per applicare i codici ANSI di colore al testo.
    
    Parametri
    ----------
    text : str
        Il testo da colorare.
    color_name : str
        Nome del colore.
    
    Restituisce
    -------
    colored_text : str
        Il testo colorato.
    
    Esempio
    -------
    >>> print(color_text("Success", "green"))
    """
    if color_name in COLORS:
        return f"{COLORS[color_name]}{text}{COLORS['reset']}"
    return text

# =============================================================================
# LETTURA DATI
# =============================================================================

def read_single_variable(filename, delimiter=None):
    """
    Legge un file con una sola colonna di dati.

    Da usare per misure ripetute di una singola grandezza fisica,
    ad esempio una serie di misure di lunghezza o di tempo.
    Le righe che iniziano con '#' vengono ignorate (commenti).

    Parametri
    ----------
    filename : str
        Percorso al file di testo.
    delimiter : str, opzionale
        Separatore di colonna. Default None (spazi/tab).

    Restituisce
    -------
    data : ndarray (1D)
        Array delle misure.

    Eccezioni
    ---------
    ValueError
        Se il file contiene più di una colonna.
    FileNotFoundError
        Se il file non esiste.

    Esempio
    -------
    >>> # File: misure_lunghezza.txt
    >>> # 10.5
    >>> # 10.3
    >>> # 10.4
    >>> data = read_single_variable("misure_lunghezza.txt")
    >>> print(data)  # [10.5, 10.3, 10.4]
    """
    data = np.loadtxt(filename, comments="#", delimiter=delimiter)
    if data.ndim != 1:
        raise ValueError("Il file deve avere una sola colonna.")
    return data


def read_xy_data(filename, delimiter=None):
    """
    Legge un file con 2 o 3 colonne: x, y, [sigma_y].

    Da usare per dati a cui si vuole applicare un fit lineare
    o studiare la relazione tra due grandezze.
    Supporta file con incertezze sulla variabile y (colonna opzionale).

    Parametri
    ----------
    filename : str
        Percorso al file di testo.
    delimiter : str, opzionale
        Separatore di colonna. Default None (spazi/tab).

    Restituisce
    -------
    x : ndarray (1D)
        Valori della variabile indipendente.
    y : ndarray (1D)
        Valori della variabile dipendente.
    sigma_y : ndarray (1D) oppure None
        Incertezze su y. None se il file ha solo 2 colonne (fit non pesato).

    Eccezioni
    ---------
    ValueError
        Se il file ha meno di 2 o più di 3 colonne.

    Esempio
    -------
    >>> # File: dati_fit.txt (con 3 colonne: x, y, sigma_y)
    >>> # 1.0  2.5  0.1
    >>> # 2.0  5.0  0.2
    >>> # 3.0  7.5  0.3
    >>> x, y, sig_y = read_xy_data("dati_fit.txt")
    >>> risultati = linear_fit_complete(x, y, sig_y)
    """
    data = np.loadtxt(filename, comments="#", delimiter=delimiter)
    if data.ndim == 1:
        raise ValueError("Il file deve avere almeno 2 colonne.")
    if data.shape[1] == 2:
        return data[:, 0], data[:, 1], None
    elif data.shape[1] == 3:
        return data[:, 0], data[:, 1], data[:, 2]
    else:
        raise ValueError("Il file deve avere 2 o 3 colonne.")


def read_multiple_variables(filename, delimiter=None, skip_header=0,
                             skip_cols=0, n_cols=None, return_type='tuple'):
    """
    Legge un file con n colonne di lunghezza non costante e non uniforme.
    Supporta file dove righe diverse hanno numero di elementi diverso.
    Le righe che iniziano con '#' vengono ignorate (commenti).

    Parametri
    ----------
    filename    : str  — percorso al file di testo.
    delimiter   : str  — separatore di colonna. Default None (spazi/tab).
    skip_header : int  — righe di intestazione da saltare. Default 0.
    skip_cols   : int  — numero di colonne iniziali da ignorare. Default 0.
    n_cols      : int  — numero di colonne da leggere dopo lo skip.
                         None = tutte quelle disponibili. Default None.
    return_type : str  — 'tuple' | 'list' | 'dict'. Default 'tuple'.

    Restituisce
    -------
    variables : tuple, list oppure dict di array numpy.

    Esempio
    -------
    File con 5 colonne: [A, B, C, D, E]

    >>> # Salta le prime 2, prendi le 2 successive → [C, D]
    >>> c, d = read_multiple_variables("dati.txt", skip_cols=2, n_cols=2)

    >>> # Salta le prime 3, prendi tutto il resto → [D, E]
    >>> d, e = read_multiple_variables("dati.txt", skip_cols=3)
    """

    if return_type not in ('tuple', 'list', 'dict'):
        raise ValueError("return_type deve essere 'tuple', 'list' o 'dict'.")

    # --- Lettura file ---
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' non trovato.")

    # --- Filtra intestazione, commenti e righe vuote ---
    data_lines = []
    for i, line in enumerate(lines):
        if i < skip_header or line.strip().startswith('#') or line.strip() == '':
            continue
        data_lines.append(line.strip())

    if not data_lines:
        raise ValueError("Il file non contiene dati validi.")

    # --- Calcola quante colonne prendere ---
    max_cols_file = max(len(line.split(delimiter)) for line in data_lines)

    if skip_cols >= max_cols_file:
        raise ValueError(
            f"skip_cols={skip_cols} >= numero di colonne nel file ({max_cols_file})."
        )

    col_start = skip_cols
    col_stop  = max_cols_file if n_cols is None else min(skip_cols + n_cols, max_cols_file)

    if n_cols is not None and skip_cols + n_cols > max_cols_file:
        import warnings
        warnings.warn(
            f"Richieste colonne [{col_start}:{skip_cols + n_cols}] ma il file ne ha "
            f"solo {max_cols_file}. Verranno lette fino alla colonna {col_stop - 1}."
        )

    n_out = col_stop - col_start
    variables = [[] for _ in range(n_out)]

    # --- Parsing riga per riga ---
    for line in data_lines:
        values = line.split(delimiter)
        for col_idx in range(col_start, min(col_stop, len(values))):
            clean_val = values[col_idx].strip().replace(',', '.')
            if clean_val == '':
                continue
            try:
                variables[col_idx - col_start].append(float(clean_val))
            except ValueError:
                raise ValueError(
                    f"Impossibile convertire '{values[col_idx]}' a float nella riga: {line}"
                )

    variables = [np.array(v) for v in variables]

    # --- Restituzione ---
    if return_type == 'dict':
        return {f'col_{col_start + i}': v for i, v in enumerate(variables)}
    elif return_type == 'list':
        return variables
    else:
        return tuple(variables)


# =============================================================================
# INCERTEZZE
# =============================================================================

def round_to_sig_figs(value, sigma, digits=2):
    """
    Arrotonda un valore e la sua incertezza alla stessa cifra significativa.
 
    Parametri
    ----------
    value : float
        Il valore da arrotondare.
    sigma : float
        L'incertezza associata al valore.
    digits : int, opzionale
        Numero di cifre significative per l'incertezza. Default 2.
 
    Restituisce
    -------
    value_rounded : str
        Il valore arrotondato formattato con zeri finali (es. "12.30").
    sigma_rounded : str
        L'incertezza arrotondata formattato con zeri finali (es. "0.67").
 
    Esempio
    -------
    >>> val, sig = round_to_sig_figs(12.3456, 0.6789)
    >>> print(f"Valore: {val}, Incertezza: {sig}")
    # Output: Valore: 12.30, Incertezza: 0.67
    
    >>> val2, sig2 = round_to_sig_figs(2.7026, 0.0031, digits=1)
    >>> print(f"Valore: {val2}, Incertezza: {sig2}")
    # Output: Valore: 2.70, Incertezza: 0.003
    """
    if sigma == 0:
        return str(value), str(sigma)
 
    # Trova l'ordine di grandezza di sigma
    order = int(np.floor(np.log10(abs(sigma))))
    
    # Numero di decimali da mostrare
    decimals = -order + digits - 1
    
    sigma_rounded = round(sigma, decimals)
    value_rounded = round(value, decimals)
 
    # Formatta come stringhe con zeri finali
    return f"{value_rounded:.{decimals}f}", f"{sigma_rounded:.{decimals}f}"

def resolution_uncertainty(instrument_res):
    """
    Calcola l'incertezza di risoluzione di uno strumento.

    Modella la risoluzione come distribuzione uniforme di ampiezza
    instrument_res, da cui sigma = res / sqrt(12).
    Tipico per strumenti digitali o meccanici a scala.

    Parametri
    ----------
    instrument_res : float
        Risoluzione dello strumento (ultima cifra significativa leggibile).
        Es: calibro digitale 0.01 mm → instrument_res = 0.01

    Restituisce
    -------
    sigma_res : float
        Incertezza di risoluzione, da passare come componente Tipo B
        a `calculate_sigma_b` e poi `get_total_uncertainty`.

    Esempio
    -------
    >>> sigma_res_calibro = resolution_uncertainty(0.01)  # calibro ±0.01 mm
    >>> sigma_res_palmer = resolution_uncertainty(0.001)  # palmer ±0.001 mm
    >>> sigma_b = calculate_sigma_b([sigma_res_calibro, sigma_offset])
    >>> result = get_total_uncertainty(mean, sigma_a, sigma_b)
    """
    return instrument_res / np.sqrt(12)


def calculate_offset(offset_measures, data, is_repeatable=False):
    """
    Calcola l'offset medio e la sua incertezza, poi corregge i dati.

    Misura l'offset come media delle misure di riferimento e lo sottrae
    ai dati reali. La stima dell'incertezza dipende dal tipo di strumento:
    - Strumenti ripetibili (es. calibro): statistica campionaria sulla media
    - Strumenti non ripetibili (es. Palmer): distribuzione uniforme
      sull'intervallo osservato, sigma = (max - min) / sqrt(12)

    Parametri
    ----------
    offset_measures : array-like
        Misure prese sullo zero o su un riferimento noto.
        Almeno 2 misure in entrambi i casi.
    data : array-like
        Misure reali da correggere.
    is_repeatable : bool, opzionale
        True  per strumenti con misure riproducibili.
        False (default) per strumenti tipo Palmer.

    Restituisce
    -------
    offset : float
        Valore medio dell'offset (= (max + min) / 2 delle misure di offset).
    sigma_offset : float
        Incertezza sull'offset. Va passata come componente Tipo B
        a `calculate_sigma_b` dopo la correzione, separata da sigma_res.
    data_corrected : ndarray
        Dati corretti: data - offset.
    info : dict
        Dizionario con dettagli utili sulla correzione:
            'N_offset'      : numero di misure di offset
            'min_offset'    : valore minimo delle misure di offset
            'max_offset'    : valore massimo delle misure di offset
            'intervallo'    : ampiezza dell'intervallo (max - min)
            'is_repeatable' : metodo usato per sigma_offset
        Da passare a `print_offset_results` per la stampa.

    Eccezioni
    ---------
    ValueError
        Se is_repeatable=True  e offset_measures contiene meno di 2 misure.
        Se is_repeatable=False e offset_measures contiene meno di 2 misure.

    Esempio
    -------
    >>> sigma_res = resolution_uncertainty(0.01)
    >>> offset_measures = [0.02, 0.01, 0.03]
    >>> misure_reali = [10.5, 10.3, 10.4]
    >>> offset, sig_off, dati_corretti, info = calculate_offset(
    ...     offset_measures, misure_reali, is_repeatable=False
    ... )
    >>> print_offset_results(offset, sig_off, info, unit="mm", color="magenta")
    >>> stats = sample_statistic(dati_corretti)
    >>> sigma_b = calculate_sigma_b([sigma_res, sig_off])
    >>> result = get_total_uncertainty(stats[0], stats[3], sigma_b)
    """
    offset_measures = np.array(offset_measures, dtype=float)
    data = np.array(data, dtype=float)
    N = len(offset_measures)

    offset = (np.max(offset_measures) + np.min(offset_measures)) / 2
    intervallo = np.max(offset_measures) - np.min(offset_measures)

    if is_repeatable and N > 1:
        # Tipo A: misure riproducibili, statistica campionaria sulla media
        sigma_offset = np.std(offset_measures, ddof=1) / np.sqrt(N)
    elif is_repeatable and N == 1:
        raise ValueError(
            "Con is_repeatable=True servono almeno 2 misure di offset "
            "per calcolare la deviazione standard."
        )
    else:
        # Non ripetibile: distribuzione uniforme sull'intervallo osservato
        if N < 2:
            raise ValueError(
                "Con is_repeatable=False servono almeno 2 misure di offset "
                "per stimare l'intervallo. Con una sola misura non è possibile "
                "stimare sigma_offset indipendentemente da sigma_res."
            )
        sigma_offset = intervallo / np.sqrt(12)

    data_corrected = data - offset

    info = {
        "N_offset":      N,
        "min_offset":    np.min(offset_measures),
        "max_offset":    np.max(offset_measures),
        "intervallo":    intervallo,
        "is_repeatable": is_repeatable,
    }

    return offset, sigma_offset, data_corrected, info


def sample_statistic(data):
    """
    Calcola le statistiche campionarie di un insieme di misure.

    Da usare per caratterizzare la distribuzione di misure ripetute
    di una stessa grandezza. Il risultato può essere passato direttamente
    a `get_total_uncertainty` per il calcolo dell'incertezza totale.

    Parametri
    ----------
    data : array-like
        Serie di misure ripetute (almeno 2).

    Restituisce
    -------
    results : tuple (mean, var, std, std_mean, N)
        mean     : media campionaria
        var      : varianza campionaria (ddof=1)
        std      : deviazione standard campionaria
        std_mean : incertezza sulla media (= std / sqrt(N)), cioè sigma_a
        N        : numero di misure

    Eccezioni
    ---------
    ValueError
        Se data contiene meno di 2 elementi.

    Esempio
    -------
    >>> data = np.array([10.5, 10.3, 10.4, 10.6, 10.2])
    >>> risultati = sample_statistic(data)
    >>> mean, var, std, std_mean, N = risultati
    >>> print(f"Media: {mean:.4f}, sigma_a: {std_mean:.4f}")
    # Output: Media: 10.4000, sigma_a: 0.1095
    >>> print_sample_statistics(mean, var, std, std_mean, N)
    """
    data = np.array(data, dtype=float)
    N = len(data)

    if N < 2:
        raise ValueError("Servono almeno 2 misure per calcolare la varianza campionaria.")

    mean = np.mean(data)
    var = np.sum((data - mean)**2) / (N - 1)
    std = np.sqrt(var)
    std_mean = std / np.sqrt(N)

    return (mean, var, std, std_mean, N)


def calculate_sigma_b(sigma_b_list):
    """
    Calcola l'incertezza di Tipo B totale sommando i contributi in quadratura.

    Ogni elemento della lista rappresenta una sorgente di incertezza sistematica
    indipendente (es. risoluzione strumento, offset, calibrazione, dilatazione
    termica). La combinazione in quadratura è valida sotto l'ipotesi che le
    sorgenti siano statisticamente indipendenti.

    Parametri
    ----------
    sigma_b_list : list of float
        Lista dei contributi Tipo B (es: [sigma_res, sigma_offset, sigma_calib]).
        Può contenere un solo elemento, oppure essere una lista vuota (restituisce 0.0).

    Restituisce
    -------
    sigma_b_tot : float
        Incertezza di Tipo B complessiva da passare a `get_total_uncertainty`.

    Esempio
    -------
    >>> sigma_res = resolution_uncertainty(0.01)
    >>> _, sigma_offset, _, _ = calculate_offset(offset_data, raw_data, is_repeatable=False)
    >>> sigma_termica = 0.05  # Es. da thermal_expansion_correction
    >>> sigma_b = calculate_sigma_b([sigma_res, sigma_offset, sigma_termica])
    >>> result = get_total_uncertainty(mean, sigma_a, sigma_b)
    """
    if np.size(sigma_b_list) == 0:
        return 0.0
    return np.sqrt(np.sum(np.array(sigma_b_list)**2))


def get_total_uncertainty(mean, sigma_a, sigma_b):
    """
    Combina l'incertezza di Tipo A e Tipo B in quadratura.
    
    Calcola l'incertezza totale come sqrt(sigma_a^2 + sigma_b^2).
    Restituisce una tupla oppure un dizionario in base al valore globale
    di RETURN_TUPLE.

    Parametri
    ----------
    mean : float
        Il valore medio delle misure.
    sigma_a : float
        Incertezza statistica (es. std_mean da sample_statistic).
    sigma_b : float
        Incertezza sistematica totale (output di calculate_sigma_b).

    Restituisce
    -------
    Se RETURN_TUPLE == True:
        tuple (mean, sigma_a, sigma_b, sigma_total)
        
    Se RETURN_TUPLE == False:
        dict con il riepilogo dei valori:
            'mean'        : valore medio
            'sigma_a'     : incertezza Tipo A
            'sigma_b'     : incertezza Tipo B
            'sigma_total' : incertezza totale combinata

    Esempio
    -------
    >>> # Con RETURN_TUPLE = True (default):
    >>> data = np.array([10.5, 10.3, 10.4])
    >>> mean, var, std, std_mean, N = sample_statistic(data)
    >>> sigma_res = resolution_uncertainty(0.01)
    >>> sigma_b = calculate_sigma_b([sigma_res])
    >>> mean, sig_a, sig_b, sig_tot = get_total_uncertainty(mean, std_mean, sigma_b)
    >>> print(f"Risultato: {mean:.4f} ± {sig_tot:.4f} mm")
    >>> 
    >>> # Oppure con RETURN_TUPLE = False (impostare in alto):
    >>> result = get_total_uncertainty(mean, std_mean, sigma_b)
    >>> print_total_uncertainty(result['mean'], result['sigma_a'],
    ...                         result['sigma_b'], result['sigma_total'], unit="mm")
    """
    sigma_total = np.sqrt(sigma_a**2 + sigma_b**2)

    if RETURN_TUPLE:
        return (mean, sigma_a, sigma_b, sigma_total)
    else:
        return {
            "mean":        mean,
            "sigma_a":     sigma_a,
            "sigma_b":     sigma_b,
            "sigma_total": sigma_total
        }

# =============================================================================
# FIT LINEARE
# =============================================================================

def linear_fit_complete(x, y, sigma_y=None):
    """
    Esegue un fit lineare pesato o non pesato: y = m*x + c

    Se sigma_y è fornita, il fit è pesato (w = 1/sigma²) e il chi quadro
    è significativo. Se sigma_y è None, il fit è non pesato e le incertezze
    sui parametri vengono stimate dal residuo (chi quadro ridotto).

    Parametri
    ----------
    x : array-like
        Valori della variabile indipendente.
    y : array-like
        Valori della variabile dipendente.
    sigma_y : array-like, opzionale
        Incertezze sui valori y. Se None, fit non pesato.

    Restituisce
    -------
    m, c : float
        Pendenza e intercetta.
    sigma_m, sigma_c : float
        Incertezze su m e c.
    cov_mc : float
        Covarianza tra m e c.
    rho : float
        Coefficiente di correlazione tra m e c (tra -1 e +1).
    chi2 : float
        Chi quadro del fit.
    chi2_rid : float
        Chi quadro ridotto (chi2 / (N-2)).
    N : int
        Numero di punti.
    weighted : bool
        True se il fit è pesato, False altrimenti.

    Eccezioni
    ---------
    ValueError
        Se N < 3, sigma_y contiene zeri, o Delta = 0.

    Esempio
    -------
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y = np.array([2.1, 4.2, 5.9, 8.0])
    >>> sigma_y = np.array([0.1, 0.1, 0.15, 0.2])
    >>> m, c, sig_m, sig_c, cov_mc, rho, chi2, chi2_rid, N, weighted = linear_fit_complete(x, y, sigma_y)
    >>> print(f"Pendenza: m = {m:.4f} ± {sig_m:.4f}")
    >>> print_fit_results(m, c, sig_m, sig_c, cov_mc, rho, chi2, chi2_rid, N, weighted)
    >>> plot_generic_fit(x, y, sigma_y, m, c, sig_m, sig_c, cov_mc)
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    N = len(x)

    if N < 3:
        raise ValueError("Servono almeno 3 punti per un fit lineare.")

    if sigma_y is None:
        sigma_y = np.ones(N)
        weighted = False
    else:
        weighted = True

    sigma_y = np.array(sigma_y, dtype=float)

    if np.any(sigma_y == 0):
        raise ValueError("Le sigma non possono essere nulle.")

    w = 1 / sigma_y**2

    S   = np.sum(w)
    Sx  = np.sum(w * x)
    Sy  = np.sum(w * y)
    Sxx = np.sum(w * x**2)
    Sxy = np.sum(w * x * y)

    Delta = S * Sxx - Sx**2

    if Delta == 0:
        raise ValueError("Delta = 0 → impossibile eseguire il fit.")

    m = (S * Sxy - Sx * Sy) / Delta
    c = (Sxx * Sy - Sx * Sxy) / Delta

    y_fit = m * x + c
    chi2 = np.sum(((y - y_fit) / sigma_y)**2)
    chi2_rid = chi2 / (N - 2)

    sigma_m2 = S / Delta
    sigma_c2 = Sxx / Delta
    cov_mc   = -Sx / Delta

    if not weighted:
        sigma_m2 *= chi2_rid
        sigma_c2 *= chi2_rid
        cov_mc   *= chi2_rid

    sigma_m = np.sqrt(sigma_m2)
    sigma_c = np.sqrt(sigma_c2)
    rho = cov_mc / (sigma_m * sigma_c)

    return (m, c, sigma_m, sigma_c, cov_mc, rho, chi2, chi2_rid, N, weighted)


def linear_fit_residuals(x, y):
    """
    Esegue un fit lineare pesato stimando sigma_y dai residui del fit.

    Metodo in tre passi:
      1. Fit non pesato per ottenere la retta di riferimento (m0, c0).
      2. Calcolo dei residui e stima di sigma empirica:
             sigma_emp = sqrt( sum(residui²) / (N-2) )
         Il ddof=2 (gradi di libertà = N-2) tiene conto dei due
         parametri m e c stimati dal fit.
      3. Fit pesato con sigma uniforme sigma_emp su tutti i punti,
         producendo un chi² che dovrebbe essere ≈ 1 per verifica.

    Utile quando le incertezze teoriche su y sono sottostimate o
    difficili da calcolare, e si vuole stimarle empiricamente dalla
    dispersione dei dati rispetto alla retta di best-fit.

    Parametri
    ----------
    x : array-like
        Valori della variabile indipendente.
    y : array-like
        Valori della variabile dipendente.

    Restituisce
    -------
    results : tuple
        Stessa struttura di `linear_fit_complete`:
        (m, c, sigma_m, sigma_c, cov_mc, rho, chi2, chi2_rid, N, weighted=True)
    sigma_emp : float
        Sigma empirica stimata dai residui (in unità di y).
        Utile da stampare o passare a `plot_generic_fit`.
    residui : ndarray
        Array dei residui y_i - (m0*x_i + c0), utile per diagnostica.

    Eccezioni
    ---------
    ValueError
        Se N < 3 (ereditato da `linear_fit_complete`).

    Esempio
    -------
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([2.1, 4.0, 6.1, 8.2, 10.0])
    >>> fit_res, sigma_emp, residui = linear_fit_residuals(x, y)
    >>> print(f"Sigma empirica dai residui: {sigma_emp:.6f}")
    >>> print_fit_results(fit_res)
    >>> m, c, sig_m, sig_c, cov_mc, *_ = fit_res
    >>> sigma_y_plot = np.full_like(x, sigma_emp)
    >>> plot_generic_fit(x, y, sigma_y_plot, m, c, sig_m, sig_c, cov_mc)
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    N = len(x)

    # Passo 1: fit non pesato
    m0, c0, *_ = linear_fit_complete(x, y, sigma_y=None)

    # Passo 2: residui e sigma empirica
    residui  = y - (m0 * x + c0)
    sigma_emp = np.sqrt(np.sum(residui**2) / (N - 2))

    # Passo 3: fit pesato con sigma uniforme
    sigma_uniform = np.full(N, sigma_emp)
    results = linear_fit_complete(x, y, sigma_y=sigma_uniform)

    return results, sigma_emp, residui


# =============================================================================
# STAMPA
# =============================================================================

def print_sample_statistics(risultati, color=None):
    """
    Stampa le statistiche campionarie in formato leggibile.

    Parametri
    ----------
    mean, var, std, std_mean, N:
        Output di `sample_statistic`.
    color : str, opzionale
        Colore dell'intestazione e del separatore ('blue', 'red',
        'green', 'yellow', 'magenta'). Default None (nessun colore).

    Esempio
    -------
    >>> data = np.array([10.5, 10.3, 10.4, 10.6])
    >>> risultati = sample_statistic(data)
    >>> mean, var, std, std_mean, N = risultati
    >>> print_sample_statistics(risultati, color="blue")
    """
    
    mean, var, std, std_mean, N = risultati
    
    header = _apply_color("\n========== STATISTICHE CAMPIONARIE ==========", color)
    body = (f"Numero misure: {N}\n"
            f"Media                     = {mean:.6f}\n"
            f"Varianza campionaria      = {var:.6f}\n"
            f"Deviazione std (misure)   = {std:.6f}\n"
            f"Sigma_a (Statistica)      = {std_mean:.6f}")
    footer = _apply_color("=============================================", color)
    print(f"{header}\n{body}\n{footer}")


def print_offset_results(offset, sigma_offset, info, unit="", color=None):
    """
    Stampa il riepilogo della correzione di offset applicata ai dati.

    Parametri
    ----------
    offset : float
        Valore dell'offset applicato.
    sigma_offset : float
        Incertezza sull'offset.
    info : dict
        Dizionario di info da `calculate_offset`.
    unit : str, opzionale
        Unità di misura. Default "".
    color : str, opzionale
        Colore dell'intestazione. Default None.

    Esempio
    -------
    >>> offset_measures = [0.02, 0.01, 0.03]
    >>> misure = [10.5, 10.3, 10.4]
    >>> offset, sigma_off, dati_corr, info = calculate_offset(offset_measures, misure, is_repeatable=False)
    >>> print_offset_results(offset, sigma_off, info, unit="mm", color="magenta")
    """
    metodo = "Tipo A (campionario)" if info["is_repeatable"] else "Tipo B (distribuzione uniforme)"
    header = _apply_color("\n============= CORREZIONE OFFSET =============", color)
    body = (f"Misure di offset usate    : {info['N_offset']}\n"
            f"Minimo offset             = {info['min_offset']:.6f} {unit}\n"
            f"Massimo offset            = {info['max_offset']:.6f} {unit}\n"
            f"Intervallo (max - min)    = {info['intervallo']:.6f} {unit}\n"
            f"Offset applicato          = {offset:.6f} {unit}\n"
            f"Sigma offset              = {sigma_offset:.6f} {unit}\n"
            f"Metodo sigma_offset       : {metodo}")
    footer = _apply_color("=============================================", color)
    print(f"{header}\n{body}\n{footer}")


def print_total_uncertainty(mean, sigma_a, sigma_b, sigma_tot, unit="", color=None):
    """
    Stampa il riepilogo delle incertezze combinate (Tipo A + Tipo B).

    Parametri
    ----------
    mean : float
        Valore medio.
    sigma_a : float
        Incertezza Tipo A.
    sigma_b : float
        Incertezza Tipo B.
    sigma_tot : float
        Incertezza totale combinata.
    unit : str, opzionale
        Unità di misura. Default "".
    color : str, opzionale
        Colore dell'intestazione. Default None.

    Esempio
    -------
    >>> # Con RETURN_TUPLE = True:
    >>> mean, sig_a, sig_b, sig_tot = get_total_uncertainty(10.4, 0.1, 0.05)
    >>> print_total_uncertainty(mean, sig_a, sig_b, sig_tot, unit="mm", color="blue")
    >>> 
    >>> # Oppure con RETURN_TUPLE = False:
    >>> result = get_total_uncertainty(10.4, 0.1, 0.05)
    >>> print_total_uncertainty(result['mean'], result['sigma_a'],
    ...                         result['sigma_b'], result['sigma_total'],
    ...                         unit="mm", color="blue")
    """
    header = _apply_color("\n========== INCERTEZZA TOTALE ==========", color)
    body = (f"Valore Medio Corretto     = {mean:.6f} {unit}\n"
            f"Incertezza Tipo A (s_a)   = {sigma_a:.6f} {unit}\n"
            f"Incertezza Tipo B (s_b)   = {sigma_b:.6f} {unit}\n"
            f"Incertezza TOTALE (s_t)   = {sigma_tot:.6f} {unit}")
    footer = _apply_color("=======================================", color)
    print(f"{header}\n{body}\n{footer}")


def print_final_result(name, mean, sigma_total, unit="", color=None):
    """
    Stampa il risultato finale in formato compatto: grandezza = valore ± sigma.

    Parametri
    ----------
    name : str
        Nome della grandezza misurata.
    mean : float
        Valore medio.
    sigma_total : float
        Incertezza totale.
    unit : str, opzionale
        Unità di misura. Default "".
    color : str, opzionale
        Colore del testo. Default None.

    Esempio
    -------
    >>> mean, sig_a, sig_b, sig_tot = get_total_uncertainty(10.4, 0.1, 0.05)
    >>> print_final_result("Lunghezza", mean, sig_tot, unit="mm", color="green")
    # Output: RISULTATO FINALE: Lunghezza = 10.400000 ± 0.111803 mm
    """
    text = f"RISULTATO FINALE: {name} = {mean:.6f} ± {sigma_total:.6f} {unit}"
    print(_apply_color(text, color))


def print_fit_results(m, c, sigma_m, sigma_c, cov_mc, rho = 0, chi2 = 0, chi2_rid = 0, N = 0, weighted = 0):
    """
    Stampa i risultati del fit lineare in formato leggibile.

    Parametri
    ----------
    results : tuple
        Output di `linear_fit_complete`.

    Esempio
    -------
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y = np.array([2.1, 4.2, 5.9, 8.0])
    >>> sig_y = np.array([0.1, 0.1, 0.15, 0.2])
    >>> m, c, sig_m, sig_c, cov_mc, rho, chi2, chi2_rid, N, weighted = linear_fit_complete(x, y, sig_y)
    >>> print_fit_results(m, c, sig_m, sig_c, cov_mc, rho, chi2, chi2_rid, N, weighted)
    """

    print("\n========== RISULTATI DEL FIT ==========")
    print(f"Numero di punti: {N}")
    print(f"Fit pesato: {'SI' if weighted else 'NO'}")
    print("----------------------------------------")
    print(f"m = {m:.6f} ± {sigma_m:.6f}")
    print(f"c = {c:.6f} ± {sigma_c:.6f}")
    print("----------------------------------------")
    print(f"Cov(m,c) = {cov_mc:.6e}")
    print(f"rho      = {rho:.6f}")
    print("----------------------------------------")
    print(f"chi2         = {chi2:.4f}")
    print(f"chi2 ridotto = {chi2_rid:.4f}")
    print("========================================\n")


# =============================================================================
# VISUALIZZAZIONE
# =============================================================================
def plot_generic_fit(x, y, sig_y, m, c, sig_m, sig_c, cov_mc=0,
                     xlabel="X", ylabel="Y", title="Fit Lineare",
                     plot=False, save=True, type="", inv=False, zoom=False, nome=""):

    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    x = np.asarray(x)
    y = np.asarray(y)
    sig_y = np.asarray(sig_y)

    x_min, x_max = min(x), max(x)
    dx = x_max - x_min
    disegno_x_min = x_min
    disegno_x_max = x_max
    
    # disegno_x_min = x_min - 0.2 * dx
    # disegno_x_max = x_max + 0.2 * dx

    x_range = np.linspace(disegno_x_min, disegno_x_max, 100000)
    y_fit = m * x_range + c
    sig_y_fit = np.sqrt((x_range**2) * (sig_m**2) + (sig_c**2) + 2 * x_range * cov_mc)

    fig, ax = plt.subplots(figsize=(8, 5))

    # ── grafico principale ────────────────────────────────────────────────────
    ax.plot(x_range, y_fit, 'r', label=f'Fit {type}: y = {m:.4f}x + {c:.4f}')
    if(inv):
        ax.fill_between(x_range,
                        y_fit - 2 * sig_y_fit,
                        y_fit + 2 * sig_y_fit,
                        color='red', alpha=0.2, label='Incertezza Fit (2$\\sigma$)')
        
    ax.errorbar(x, y, yerr=sig_y, fmt='o', markersize=2,
                color='#1F4E78', capsize=2, capthick=1.5,
                elinewidth=1.5, label='Dati Sperimentali',
                mec='darkblue', mew=1.2, zorder=5)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    if(zoom):
        # ── punto su cui zoomare ─────────────────────────────────────────────────
        idx      = np.argmin(np.abs(x - np.median(x)))
        xz       = x[idx]
        yz_data  = y[idx]
        yz_fit   = m * xz + c
        sig_band = np.sqrt(xz**2 * sig_m**2 + sig_c**2 + 2 * xz * cov_mc)

        # ── calcolo del box quadrato in coordinate dati ───────────────────────────
        # Recupera le dimensioni reali degli assi in pixel dopo il layout
        renderer  = fig.canvas.get_renderer()
        bbox_ax   = ax.get_window_extent(renderer=renderer)
        ax_w_px   = bbox_ax.width
        ax_h_px   = bbox_ax.height

        xl = ax.get_xlim()
        yl = ax.get_ylim()
        x_per_px = (xl[1] - xl[0]) / ax_w_px   # unità dato per pixel in x
        y_per_px = (yl[1] - yl[0]) / ax_h_px   # unità dato per pixel in y

        # Partiamo da half_y (deve contenere errore + banda)
        half_y = max(sig_y[idx], 2 * sig_band) * 4.0
        # half_x calcolato per avere lo stesso numero di pixel in x e in y → box quadrato
        half_x = half_y * (x_per_px / y_per_px)

        x1, x2 = xz - half_x, xz + half_x
        y1, y2 = min(yz_data, yz_fit) - half_y, max(yz_data, yz_fit) + half_y

        # ── inset (quadrato: stessa percentuale width e height) ───────────────────
        axins = ax.inset_axes([0.6, 0.1, 0.38, 0.38])

        mask = (x_range >= x1) & (x_range <= x2)
        axins.plot(x_range[mask], y_fit[mask], 'r', lw=1.5)
        axins.fill_between(x_range[mask],
                        (y_fit - 2 * sig_y_fit)[mask],
                        (y_fit + 2 * sig_y_fit)[mask],
                        color='red', alpha=0.2)
        axins.errorbar([xz], [yz_data], yerr=[sig_y[idx]],
                    fmt='o', markersize=5, color='#1F4E78',
                    capsize=4, capthick=1.5, elinewidth=1.5,
                    mec='darkblue', mew=1.2, zorder=5)

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.tick_params(labelsize=7)
        axins.grid(True, linestyle='--', alpha=0.5)
        axins.set_title("Zoom", fontsize=8, pad=3)

        mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.4", lw=0.8, ls="--")

    if save:
        plt.savefig(f"fit_{nome}.jpg", dpi=1000, bbox_inches='tight')
    showOrClose(plot)


def scatter_pre_fit(x, y, sigma_y, xlabel="X", ylabel="Y", title="Scatter dati", ax=None):
    """
    Visualizza i dati sperimentali su un oggetto Axes specifico.
    """
    # ax.scatter(x, y, label='Media dei dati sperimentali', marker='d', 
    #           color='orange', s=2, zorder=10)
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(x, y, yerr=sigma_y, fmt='o', markersize=3, 
                capsize=0.5, capthick=0.5,
                elinewidth=1.5, label='Dati Sperimentali',
                mec='darkblue', mew=1.2, zorder=5)    
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
def plot_normal(mean, sigma, label=None, color=None, unit=""):
    """
    Disegna una distribuzione normale (gaussiana) sul grafico corrente (axes attivo).

    Utile per visualizzare la distribuzione di probabilità associata
    a una misura con incertezza gaussiana.

    Parametri
    ----------
    mean : float
        Media della distribuzione (μ).
    sigma : float
        Deviazione standard della distribuzione (σ).
    label : str, opzionale
        Etichetta da mostrare nella legenda. Se None, mostra "μ = ... σ = ...".
    color : str, opzionale
        Colore della curva e dell'area. Default None (usa colore default).
    unit : str, opzionale
        Unità di misura per le etichette. Default "".

    Esempio
    -------
    >>> plt.figure(figsize=(10, 6))
    >>> mean, sig_a, sig_b, sig_tot = get_total_uncertainty(10.4, 0.1, 0.05)
    >>> plot_normal(mean, sig_tot, label="Lunghezza", color="blue", unit="mm")
    >>> plt.xlabel("Lunghezza [mm]")
    >>> plt.ylabel("Densità di probabilità")
    >>> plt.show()
    """
    from scipy.stats import norm as _norm

    x = np.linspace(mean - 4*sigma, mean + 4*sigma, 10000)
    y = _norm.pdf(x, mean, sigma)

    if label is None:
        label = f"μ = {mean:.5f} {unit}  σ = {sigma:.5f} {unit}"
    else:
        label = label + ": " + f"μ = {mean:.5f} {unit}  σ = {sigma:.5f} {unit}"

    plt.plot(x, y, color=color, label=label)
    plt.fill_between(x, y, color=color, alpha=0.1)

    
def numeric_derivatives(dati, tempo, passo=1):
    """
    Calcola la derivata a coppie:
    Se passo=1: (x1-x0)/(t1-t0), (x2-x1)/(t2-t1)...
    Se passo=2: (x1-x0)/(t1-t0), (x3-x2)/(t3-t2), (x5-x4)/(t5-t4)...
    """
    risultati = []
    tempi_res = []
    
    n = min(len(dati), len(tempo))
    
    # range(start, stop, step)
    # Partiamo da 0, arriviamo alla fine, e saltiamo di 2 in 2
    for i in range(0, n - 1, passo):
        # Calcoliamo la differenza tra il punto i+1 e il punto i
        dy = dati[i + 1] - dati[i]
        dt = tempo[i + 1] - tempo[i]
        
        risultati.append(dy / dt)
        # Punto medio tra i due punti della coppia
        tempi_res.append((tempo[i + 1] + tempo[i]) / 2)

    return np.array(tempi_res), np.array(risultati)

def numeric_derivatives_uncertainty(values, delta, sigma_values, sigma_delta, passo=1):
    values = np.array(values, dtype=float)
    n = len(values)

    if np.isscalar(sigma_values):
        sigma_values = np.full(n, sigma_values)
    else:
        sigma_values = np.array(sigma_values, dtype=float)

    sigma_deriv = []
    for i in range(0, n - 1, passo):
        delta_val = values[i + 1] - values[i]
        sv = np.sqrt((1 / delta)**2 * (sigma_values[i + 1]**2 + sigma_values[i]**2) + (delta_val / delta**2)**2 * sigma_delta**2)
        sigma_deriv.append(sv)

    return np.array(sigma_deriv)

def find_motion_indices(t, x, soglia_pos=0.002, finestra=10, inizio_ricerca=0.5, plot=True, 
                        save=False, name="motion_indices.png", write=True):
    """
    Trova gli indici di inizio e fine movimento basandosi sulla posizione.

    Parametri
    ----------
    t, x            : array di tempi e posizioni grezze
    soglia_pos      : variazione minima/massima (m) per stabilire se il carrello è fermo
    finestra        : numero di campioni consecutivi da controllare
    inizio_ricerca  : float (0.0 - 1.0), frazione da cui cercare lo stop. Default 0.5.

    Restituisce
    -------
    i_inizio, i_stop : int   indici tra cui il carrello è in moto
    """
    meta = int(len(x) * inizio_ricerca)

    # --- Inizio movimento: ultima finestra FERMA nella prima metà ---
    i_inizio = 0
    for i in range(meta - finestra, 0, -1):  # scorre al contrario
        if (np.max(x[i:i + finestra]) - np.min(x[i:i + finestra])) < soglia_pos:
            i_inizio = i + finestra  # parte subito dopo l'ultima finestra ferma
            break

    # --- Fine movimento: prima finestra sotto soglia (cerca nella seconda metà) ---
    i_stop = len(x) - 1
    for i in range(meta, len(x) - finestra):
        if (np.max(x[i:i + finestra]) - np.min(x[i:i + finestra])) < soglia_pos:
            i_stop = i
            break

    # --- Conferma visiva ---
    if(plot or save):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, x, color='steelblue', lw=1, label='Posizione')
        ax.scatter(t, x, color='blue', edgecolor='black', lw=0.4, s=10, label='Posizione')
        ax.axvline(t[i_inizio], color='green', linestyle='--', label=f'Inizio moto (t={t[i_inizio]:.3f}s)')
        ax.axvline(t[i_stop],   color='red',   linestyle='--', label=f'Fine moto   (t={t[i_stop]:.3f}s)')
        ax.axvspan(t[0],        t[i_inizio], color='gray',  alpha=0.15, label='Tagliato')
        ax.axvspan(t[i_stop],   t[-1],       color='gray',  alpha=0.15)
        ax.set_ylabel('Posizione [m]')
        ax.set_xlabel('Tempo [s]')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.suptitle('Verifica tratto di moto — chiudi la finestra per confermare')
        plt.tight_layout()
    if(save):
        plt.savefig(name, dpi=300)
    
    if plot: plt.show()
    else: plt.close()

    if write:
        print(f"Inizio moto : t = {t[i_inizio]:.4f} s  →  indice {i_inizio}")
        print(f"Fine moto   : t = {t[i_stop]:.4f} s  →  indice {i_stop}")
        
        print(f"Campioni utili: {i_stop - i_inizio} su {len(t)} totali")
    return i_inizio, i_stop

def showOrClose(plot):
    if plot:
        plt.show()
    else:
        plt.close()

def findBounce(t, pos):
    # Massimi locali
    idx_max, _ = find_peaks(pos, prominence=0.03, distance=10)
    
    # Minimi locali (invertiamo il segnale)
    idx_min, _ = find_peaks(-pos, prominence=0.03, distance=10)

    # Uniamo e ordiniamo per tempo
    idx_all = np.sort(np.concatenate([idx_max, idx_min]))
    
    times = t[idx_all]
    positions = pos[idx_all]
    types = ['max' if i in idx_max else 'min' for i in idx_all]

    return times, positions, types

def find_all_still_segments(t, x, soglia_pos=0.002, finestra=10,
                             durata_minima=20, plot=True, save=False,
                             name="still_segments.png", write=True):
    """
    Trova tutti i tratti in cui l'oggetto è fermo (a meno di fluttuazioni).

    Parametri
    ----------
    t, x            : array di tempi e posizioni grezze
    soglia_pos      : variazione massima (m) entro cui la finestra è considerata ferma
    finestra        : numero di campioni per ogni finestra di controllo
    durata_minima   : numero minimo di campioni consecutivi fermi per formare un segmento
    plot            : bool, mostra il grafico
    save            : bool, salva il grafico
    name            : str, nome file output

    Restituisce
    -------
    segmenti : list of (i_inizio, i_stop)
        Lista di coppie di indici, uno per ogni tratto fermo trovato.
    """

    # --- 1. Mappa booleana: True dove la finestra è ferma ---
    n = len(x)
    is_still = np.zeros(n, dtype=bool)

    for i in range(n - finestra):
        if (np.max(x[i:i + finestra]) - np.min(x[i:i + finestra])) < soglia_pos:
            is_still[i:i + finestra] = True

    # --- 2. Raggruppa campioni contigui fermi in segmenti ---
    segmenti = []
    in_segment = False

    for i in range(n):
        if is_still[i] and not in_segment:
            inizio = i
            in_segment = True
        elif not is_still[i] and in_segment:
            fine = i - 1
            in_segment = False
            if (fine - inizio) >= durata_minima:
                segmenti.append((inizio, fine))

    # Chiude l'ultimo segmento se il segnale finisce mentre si è fermi
    if in_segment:
        fine = n - 1
        if (fine - inizio) >= durata_minima:
            segmenti.append((inizio, fine))

    # --- 3. Conferma visiva ---
    if plot or save:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, x, color='steelblue', lw=1, label='Posizione')
        ax.scatter(t, x, color='blue', edgecolor='black', lw=0.4, s=8)

        colors = plt.cm.Set2.colors  # palette di colori distinti per i segmenti

        for idx, (i0, i1) in enumerate(segmenti):
            col = colors[idx % len(colors)]
            ax.axvspan(t[i0], t[i1], alpha=0.35, color=col,
                       label=f'Quiete {idx+1}: [{t[i0]:.3f} – {t[i1]:.3f}] s')
            ax.axvline(t[i0], color=col, linestyle='--', lw=1)
            ax.axvline(t[i1], color=col, linestyle=':',  lw=1)

        ax.set_ylabel('Posizione [m]')
        ax.set_xlabel('Tempo [s]')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.suptitle(f'Tratti fermi trovati: {len(segmenti)}')
        plt.tight_layout()

        if save:
            plt.savefig(name, dpi=300)
        if plot:
            plt.show()
        else:
            plt.close()

    if write:
        # --- 4. Report testuale ---
        print(f"Tratti fermi trovati: {len(segmenti)}")
        for idx, (i0, i1) in enumerate(segmenti):
            print(f"  Quiete {idx+1}: t = [{t[i0]:.4f} – {t[i1]:.4f}] s  "
                f"→  indici [{i0} – {i1}]  ({i1-i0} campioni)")

    return segmenti

def find_direction_changes(t, pos, prominenza=0.03, distanza=10,
                            plot=True, save=False, name="direction_changes.png", write=True):
    """
    Trova tutti i cambi di direzione di un corpo in caduta (rimbalzi).

    Identifica massimi e minimi locali della posizione, che corrispondono
    agli istanti in cui la velocità si annulla e inverte segno.
    Sostituisce la funzione `findBounce` con un'implementazione più completa.

    Parametri
    ----------
    t         : array  — tempi [s]
    pos       : array  — posizione [m]
    prominenza : float — prominenza minima del picco (filtra i falsi positivi).
                         Default 0.03. Aumentare se ci sono oscillazioni di rumore.
    distanza  : int    — distanza minima in campioni tra due picchi consecutivi.
                         Default 10.
    plot      : bool   — mostra il grafico. Default True.
    save      : bool   — salva il grafico. Default False.
    name      : str    — nome del file di output. Default "direction_changes.png".

    Restituisce
    -------
    eventi : list of dict
        Lista ordinata per tempo. Ogni dict contiene:
            'indice'    : int    — indice nel array originale
            'tempo'     : float  — istante del cambio di direzione [s]
            'posizione' : float  — posizione al cambio [m]
            'tipo'      : str    — 'max' (sale → scende) | 'min' (scende → sale)

    Esempio
    -------
    >>> eventi = find_direction_changes(t, pos, prominenza=0.05)
    >>> for e in eventi:
    ...     print(f"{e['tipo']:3s}  t = {e['tempo']:.4f} s   pos = {e['posizione']:.4f} m")
    """

    # --- 1. Trova massimi e minimi locali ---
    idx_max, prop_max = find_peaks( pos, prominence=prominenza, distance=distanza)
    idx_min, prop_min = find_peaks(-pos, prominence=prominenza, distance=distanza)

    # --- 2. Costruisci lista eventi unificata ---
    eventi = []
    for i in idx_max:
        eventi.append({'indice': i, 'tempo': t[i], 'posizione': pos[i], 'tipo': 'max'})
    for i in idx_min:
        eventi.append({'indice': i, 'tempo': t[i], 'posizione': pos[i], 'tipo': 'min'})

    eventi.sort(key=lambda e: e['indice'])

    # --- 3. Conferma visiva ---
    if plot or save:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, pos, color='steelblue', lw=1, label='Posizione', zorder=1)

        t_max = [e['tempo']    for e in eventi if e['tipo'] == 'max']
        p_max = [e['posizione'] for e in eventi if e['tipo'] == 'max']
        t_min = [e['tempo']    for e in eventi if e['tipo'] == 'min']
        p_min = [e['posizione'] for e in eventi if e['tipo'] == 'min']

        ax.scatter(t_max, p_max, color='red',   edgecolor='darkred',  s=60,
                   zorder=5, label=f'Massimi (sale → scende)  n={len(t_max)}')
        ax.scatter(t_min, p_min, color='green', edgecolor='darkgreen', s=60,
                   marker='v', zorder=5, label=f'Minimi  (scende → sale) n={len(t_min)}')

        for e in eventi:
            ax.axvline(e['tempo'], color='red' if e['tipo'] == 'max' else 'green',
                       linestyle='--', lw=0.7, alpha=0.5)

        ax.set_xlabel('Tempo [s]')
        ax.set_ylabel('Posizione [m]')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.suptitle(f'Cambi di direzione trovati: {len(eventi)}  '
                     f'(max: {len(t_max)}, min: {len(t_min)})')
        plt.tight_layout()

        if save:
            plt.savefig(name, dpi=300)
        if plot:
            plt.show()
        else:
            plt.close()

    if write:
        # --- 4. Report testuale ---
        print(f"Cambi di direzione trovati: {len(eventi)}")
        for k, e in enumerate(eventi):
            simbolo = '▲ MAX' if e['tipo'] == 'max' else '▼ MIN'
            print(f"  [{k+1}] {simbolo}  t = {e['tempo']:.4f} s   "
                f"pos = {e['posizione']:.6f} m   (indice {e['indice']})")

    return eventi

def find_linear_range(x, y, sigma_y=None, window=10, step=1,
                      soglia_chi2=2.0, durata_minima=10,
                      plot=True, save=False, name="linear_range.png",
                      write=True):
    """
    Trova il range di linearità nei dati usando un fit su finestra scorrevole.

    Per ogni finestra di `window` punti calcola un fit lineare e registra
    il chi² ridotto. Le finestre con chi²_rid < soglia_chi2 sono considerate
    "lineari". Viene poi selezionato il tratto contiguo più lungo.

    Se sigma_y è None, il fit usa i residui per stimare le incertezze
    (via linear_fit_residuals), rendendo il chi²_rid confrontabile tra
    finestre diverse ma non interpretabile in senso assoluto: in quel caso
    abbassare soglia_chi2 (es. 1.2–1.5).

    Parametri
    ----------
    x, y      : array-like
        Dati sperimentali.
    sigma_y   : array-like oppure None
        Incertezze su y. Se None il fit è non pesato.
    window    : int
        Numero di punti per finestra scorrevole. Default 10.
    step      : int
        Passo di scorrimento della finestra. Default 1.
    soglia_chi2 : float
        Soglia massima su chi²_rid per considerare un punto "lineare".
        Default 2.0.
    durata_minima : int
        Minimo numero di punti contigui per formare un segmento lineare.
        Default 10.
    plot      : bool   — mostra il grafico. Default True.
    save      : bool   — salva il grafico. Default False.
    name      : str    — nome file di output. Default "linear_range.png".

    Restituisce
    -------
    i_start, i_stop : int
        Indici (inclusivi) del tratto più lungo con chi²_rid < soglia_chi2.
    chi2_map : ndarray, shape (N,)
        Mappa del chi²_rid medio su ogni punto originale (NaN fuori copertura).

    Esempio
    -------
    >>> i0, i1, chi2_map = find_linear_range(x, y, sigma_y,
    ...                                       window=15, soglia_chi2=2.0)
    >>> x_lin, y_lin = x[i0:i1+1], y[i0:i1+1]
    >>> sig_lin = sigma_y[i0:i1+1] if sigma_y is not None else None
    >>> print(f"Range lineare: indici [{i0} – {i1}]  x ∈ [{x[i0]:.4f}, {x[i1]:.4f}]")
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(x)

    if sigma_y is not None:
        sigma_y = np.asarray(sigma_y, dtype=float)

    # --- 1. Fit su finestra scorrevole → chi²_rid per finestra ---------------
    chi2_rid_windows = np.full(N, np.nan)   # valore di chi²_rid centrato sul mid-point
    counts           = np.zeros(N)          # quante finestre coprono ogni punto

    # accumulo per media pesata
    chi2_acc = np.zeros(N)

    for i in range(0, N - window + 1, step):
        xi = x[i : i + window]
        yi = y[i : i + window]
        syi = sigma_y[i : i + window] if sigma_y is not None else None

        try:
            if syi is not None:
                *_, chi2_rid, _, _ = linear_fit_complete(xi, yi, syi)
            else:
                (_, _, _, _, _, _, _, chi2_rid, _, _), _, _ = linear_fit_residuals(xi, yi)
        except (ValueError, ZeroDivisionError):
            continue

        # Distribuisce il chi²_rid su tutti i punti della finestra
        chi2_acc[i : i + window] += chi2_rid
        counts[i : i + window]   += 1

    # Media per ogni punto
    valid = counts > 0
    chi2_map = np.full(N, np.nan)
    chi2_map[valid] = chi2_acc[valid] / counts[valid]

    # --- 2. Mappa booleana: punto "lineare" se chi²_rid medio < soglia -------
    is_linear = np.zeros(N, dtype=bool)
    is_linear[valid] = chi2_map[valid] < soglia_chi2

    # --- 3. Raggruppa tratti lineari contigui ---------------------------------
    segmenti = []
    in_seg   = False

    for i in range(N):
        if is_linear[i] and not in_seg:
            inizio = i
            in_seg = True
        elif not is_linear[i] and in_seg:
            fine = i - 1
            in_seg = False
            if (fine - inizio + 1) >= durata_minima:
                segmenti.append((inizio, fine))
    if in_seg:
        fine = N - 1
        if (fine - inizio + 1) >= durata_minima:
            segmenti.append((inizio, fine))

    if not segmenti:
        print(_apply_color(
            f"Nessun tratto lineare trovato con chi²_rid < {soglia_chi2} "
            f"e durata ≥ {durata_minima} punti. Prova ad aumentare soglia_chi2.",
            "red"))
        return None, None, chi2_map

    # Tratto più lungo
    best = max(segmenti, key=lambda s: s[1] - s[0])
    i_start, i_stop = best
    
    if write:
        # --- 4. Report testuale --------------------------------------------------
        header = _apply_color("\n========== RANGE DI LINEARITÀ ==========", "green")
        print(header)
        print(f"Finestra scorrevole : {window} punti  |  soglia chi²_rid : {soglia_chi2}")
        print(f"Tratti lineari trovati: {len(segmenti)}")
        for k, (i0, i1) in enumerate(segmenti):
            tag = " ← selezionato" if (i0, i1) == best else ""
            print(f"  [{k+1}] indici [{i0} – {i1}]  "
                f"x ∈ [{x[i0]:.4f}, {x[i1]:.4f}]  "
                f"({i1-i0+1} punti){tag}")
        print(_apply_color(f"Range selezionato: indici [{i_start} – {i_stop}]  "
                        f"x ∈ [{x[i_start]:.4f}, {x[i_stop]:.4f}]", "green"))
        print(_apply_color("=========================================", "green"))

    # --- 5. Grafico ----------------------------------------------------------
    if plot or save:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                        gridspec_kw={'height_ratios': [2, 1]},
                                        sharex=True)

        # -- pannello superiore: dati + range selezionato --
        if sigma_y is not None:
            ax1.errorbar(x, y, yerr=sigma_y, fmt='o', markersize=3,
                         color='#1F4E78', capsize=4, capthick=1.2,
                         elinewidth=1.2, label='Dati', mec='darkblue', mew=1)
        else:
            ax1.scatter(x, y, s=15, color='#1F4E78', label='Dati', zorder=3)

        ax1.axvspan(x[i_start], x[i_stop], color='green', alpha=0.15,
                    label=f'Range lineare [{x[i_start]:.3f} – {x[i_stop]:.3f}]')
        ax1.axvline(x[i_start], color='green', linestyle='--', lw=1.2)
        ax1.axvline(x[i_stop],  color='green', linestyle='--', lw=1.2)

        # Mostra anche gli altri segmenti in grigio
        for (i0, i1) in segmenti:
            if (i0, i1) != best:
                ax1.axvspan(x[i0], x[i1], color='orange', alpha=0.12)

        ax1.set_ylabel("y")
        ax1.legend(fontsize=8)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_title("Dati e range di linearità")

        # -- pannello inferiore: mappa chi²_rid --
        x_valid = x[valid]
        ax2.plot(x_valid, chi2_map[valid], color='steelblue', lw=1.5,
                 label='chi² rid. medio (finestra scorrevole)')
        ax2.axhline(soglia_chi2, color='red', linestyle='--', lw=1,
                    label=f'Soglia = {soglia_chi2}')
        ax2.axvspan(x[i_start], x[i_stop], color='green', alpha=0.15)
        ax2.set_ylabel("chi² rid. medio")
        ax2.set_xlabel("x")
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        if save:
            plt.savefig(name, dpi=150, bbox_inches='tight')
        showOrClose(plot)

    return i_start, i_stop, chi2_map

def covariance(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    cov = 1 / len(x) * (np.sum(x - x_m) * np.sum(y - y_m))
    return cov
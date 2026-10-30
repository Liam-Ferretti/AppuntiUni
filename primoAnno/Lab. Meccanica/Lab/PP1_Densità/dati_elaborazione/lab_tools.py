"""
lab_tools.py
============
Libreria di utility per il laboratorio di meccanica.

Funzioni disponibili
--------------------
Lettura dati:
    read_single_variable(filename)      Legge un file con una sola colonna
    read_xy_data(filename)              Legge un file con 2 o 3 colonne (x, y, sigma_y)

Incertezze:
    resolution_uncertainty(res)         Calcola sigma di risoluzione (res/sqrt(12))
    calculate_offset(...)               Corregge i dati sottraendo l'offset e stima sigma_offset
    sample_statistic(data)              Calcola media, varianza, std su misure ripetute
    get_total_uncertainty(stats, ...)   Combina incertezza Tipo A e Tipo B in quadratura

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
1. read_single_variable / read_xy_data
2. resolution_uncertainty
3. calculate_offset           (se necessario)
4. print_offset_results       (se necessario)
5. sample_statistic
6. get_total_uncertainty
7. print_sample_statistics / print_total_uncertainty / print_final_result
8. linear_fit_complete        (se necessario)
9. print_fit_results          (se necessario)
10. plot_generic_fit           (se necessario)

Dipendenze
----------
- numpy
- matplotlib

Autore: Liam Ferretti
Versione: 1.2
"""

import numpy as np
import matplotlib.pyplot as plt

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
    """Funzione interna per applicare i codici ANSI."""
    if color_name in COLORS:
        return f"{COLORS[color_name]}{text}{COLORS['reset']}"
    return text

def color_text(text, color_name):
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

    Parametri
    ----------
    filename : str
        Percorso al file di testo. Le righe che iniziano con '#'
        vengono ignorate (commenti).
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

    Esempio
    -------
    >>> data = read_single_variable("misure_lunghezza.txt")
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

    Parametri
    ----------
    filename : str
        Percorso al file di testo. Le righe che iniziano con '#'
        vengono ignorate (commenti).
    delimiter : str, opzionale
        Separatore di colonna. Default None (spazi/tab).

    Restituisce
    -------
    x : ndarray (1D)
    y : ndarray (1D)
    sigma_y : ndarray (1D) oppure None
        None se il file ha solo 2 colonne (fit non pesato).

    Eccezioni
    ---------
    ValueError
        Se il file ha meno di 2 o più di 3 colonne.

    Esempio
    -------
    >>> x, y, sig_y = read_xy_data("dati_fit.txt")
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
        Numero di cifre significative. Default 2.
 
    Restituisce
    -------
    value_rounded : str
        Il valore arrotondato formattato con zeri finali (es. "12.30").
    sigma_rounded : str
        L'incertezza arrotondata formattata con zeri finali (es. "0.67").
 
    Esempio
    -------
    >>> val, sig = round_to_sig_figs(12.3456, 0.6789)
    >>> print(f"Valore: {val}, Incertezza: {sig}")
    Valore: 12.30, Incertezza: 0.67
    
    >>> val2, sig2 = round_to_sig_figs(2.7026, 0.0031)
    >>> print(f"Valore: {val2}, Incertezza: {sig2}")
    Valore: 2.703, Incertezza: 0.003
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

def thermal_expansion_correction(data, alpha, delta_T, sigma_delta_T=0.0):
    """
    Corregge i dati tenendo conto della dilatazione termica lineare e
    stima l'incertezza introdotta dalla correzione.

    Usa la formula L_corr = L_misurato - (L_misurato * alpha * delta_T)
    per riportare la misura alla temperatura di riferimento.

    Parametri
    ----------
    data : array-like
        Misure da correggere (es. lunghezze).
    alpha : float
        Coefficiente di dilatazione termica lineare dell'oggetto (es. in K^-1 o °C^-1).
    delta_T : float
        Variazione di temperatura (T_misura - T_riferimento).
    sigma_delta_T : float, opzionale
        Incertezza sulla misurazione della temperatura. Default 0.0.

    Restituisce
    -------
    data_corrected : ndarray
        Array dei dati corretti per la dilatazione termica.
    sigma_thermal : float
        Incertezza sistematica (Tipo B) dovuta alla correzione termica, calcolata
        tramite propagazione degli errori. Da aggiungere a `sigma_b_list`.

    Esempio
    -------
    >>> alpha_alluminio = 23e-6
    >>> delta_T = 5.0 # misurato a 25°C invece di 20°C
    >>> dati_corretti, sigma_termica = thermal_expansion_correction(
    ...     misure, alpha_alluminio, delta_T, sigma_delta_T=0.5
    ... )
    >>> sigma_b_tot = calculate_sigma_b([sigma_res, sigma_offset, sigma_termica])
    """
    data = np.array(data, dtype=float)

    # Calcolo della correzione
    delta_L = data * alpha * delta_T
    data_corrected = data - delta_L

    # Propagazione dell'incertezza sulla correzione (Tipo B)
    # Calcolata sul valore medio per avere un'unica sigma_b rappresentativa
    L_mean = np.mean(data)

    # Derivate parziali per la propagazione della varianza:
    # d(Delta_L)/d(alpha) = L_mean * delta_T
    # d(Delta_L)/d(delta_T) = L_mean * alpha
    var_T_term = (L_mean * alpha * sigma_delta_T)**2

    sigma_thermal = np.sqrt(var_T_term)

    return data_corrected, sigma_thermal


def resolution_uncertainty(instrument_res):
    """
    Calcola l'incertezza di risoluzione di uno strumento.

    Modella la risoluzione come distribuzione uniforme di ampiezza
    instrument_res, da cui sigma = res / sqrt(12).

    Parametri
    ----------
    instrument_res : float
        Risoluzione dello strumento (ultima cifra significativa leggibile).
        Es: calibro digitale 0.01 mm → instrument_res = 0.01

    Restituisce
    -------
    sigma_res : float
        Incertezza di risoluzione, da passare come componente Tipo B
        a `get_total_uncertainty`.

    Esempio
    -------
    >>> sigma_res = resolution_uncertainty(0.01)
    >>> result = get_total_uncertainty(stats, sigma_b_list=[sigma_res, sigma_offset])
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
        a `get_total_uncertainty` dopo la correzione, separata da sigma_res.
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
    >>> offset, sig_off, dati_corretti, info = calculate_offset(
    ...     offset_measures, misure_reali, is_repeatable=False
    ... )
    >>> print_offset_results(offset, sig_off, info, color="magenta")
    >>> result = get_total_uncertainty(
    ...     sample_statistic(dati_corretti),
    ...     sigma_b_list=[sigma_res, sig_off]
    ... )
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
    a `get_total_uncertainty` per il calcolo dell'incertezza totale,
    evitando di ricalcolare la media due volte.

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
        std_mean : incertezza sulla media (std / sqrt(N))
        N        : numero di misure

    Eccezioni
    ---------
    ValueError
        Se data contiene meno di 2 elementi.

    Esempio
    -------
    >>> stats = sample_statistic(data)
    >>> print_sample_statistics(stats)
    >>> result = get_total_uncertainty(stats, sigma_b_list=[sigma_res, sigma_offset])
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
        Può contenere un solo elemento.

    Restituisce
    -------
    sigma_b_tot : float
        Incertezza di Tipo B complessiva da passare a `get_total_uncertainty`.

    Esempio
    -------
    >>> sigma_res = resolution_uncertainty(0.01)
    >>> _, sigma_offset, _, _ = calculate_offset(offset_data, raw_data)
    >>> sigma_b = calculate_sigma_b([sigma_res, sigma_offset])
    >>> result = get_total_uncertainty(mean, sigma_a, sigma_b)
    """
    if np.size(sigma_b_list) == 0:
        return 0.0
    return np.sqrt(np.sum(np.array(sigma_b_list)**2))


def get_total_uncertainty(mean, sigma_a, sigma_b):
    """
    Combina l'incertezza di Tipo A e Tipo B in quadratura.

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
    dict con il riepilogo dei valori:
        'mean'        : valore medio
        'sigma_a'     : incertezza Tipo A
        'sigma_b'     : incertezza Tipo B
        'sigma_total' : incertezza totale combinata

    Esempio
    -------
    >>> stats = sample_statistic(data_corrected)
    >>> sigma_b = calculate_sigma_b([sigma_res, sigma_offset])
    >>> result = get_total_uncertainty(stats[0], stats[3], sigma_b)
    >>> print_total_uncertainty(result['mean'], result['sigma_a'],
    ...                         result['sigma_b'], result['sigma_total'], unit="mm")
    """
    sigma_total = np.sqrt(sigma_a**2 + sigma_b**2)

    return {
        "mean":        mean,
        "sigma_a":     sigma_a,
        "sigma_b":     sigma_b,
        "sigma_total": sigma_total
    }


# =============================================================================
# DENSITÀ
# =============================================================================

def calculate_density(height, diameter, mass,
                  sigma_height=None, sigma_diameter=None, sigma_mass=None,
                  to_cm3=True):
    """
    Calcola la densità di un cilindro per ogni terna (h, d, m) e propaga
    le incertezze tramite derivate parziali analitiche.

    Formula: rho = 4*m / (pi * d^2 * h)

    Propagazione degli errori (grandezze indipendenti):
        sigma_rho / rho = sqrt( (sigma_m/m)^2
                            + (2*sigma_d/d)^2
                            + (sigma_h/h)^2 )

    Parametri
    ----------
    height : array-like o float
        Misure di altezza del cilindro in mm (stessa unità di diameter).
    diameter : array-like o float
        Misure di diametro del cilindro in mm.
    mass : array-like o float
        Misure di massa del cilindro in g.
    sigma_height : array-like or float, opzionale
        Incertezze sull'altezza. Può essere uno scalare (applicato a tutti)
        o un array di lunghezza N. Default None (propagazione non calcolata).
    sigma_diameter : array-like or float, opzionale
        Incertezze sul diametro. Default None.
    sigma_mass : array-like or float, opzionale
        Incertezze sulla massa. Default None.
    to_cm3 : bool, opzionale
        Se True (default), converte il risultato da g/mm³ a g/cm³
        moltiplicando per 1000 (poiché 1 cm³ = 1000 mm³).
        Impostare a False solo se height e diameter sono già in cm.

    Restituisce
    -------
    densities : ndarray, shape (N,)
        Array delle N densità calcolate in g/cm³ (se to_cm3=True)
        oppure g/mm³ (se to_cm3=False).
    sigma_densities : ndarray, shape (N,) oppure None
        Incertezze sulle singole densità propagate analiticamente,
        nelle stesse unità di densities.
        None se nessuna sigma è stata fornita.

    Eccezioni
    ---------
    ValueError
        Se gli array height, diameter e mass non hanno la stessa lunghezza.
        Se sigma_* è un array ma ha lunghezza diversa da N.

    Esempio
    -------
    >>> sigma_h = resolution_uncertainty(0.01)   # es. calibro in mm
    >>> sigma_d = resolution_uncertainty(0.01)
    >>> sigma_m = resolution_uncertainty(0.001)  # es. bilancia in g
    >>> densities, sigma_rho = calculate_density(
    ...     height, diameter, mass,
    ...     sigma_height=sigma_h, sigma_diameter=sigma_d, sigma_mass=sigma_m
    ... )
    >>> # densities è in g/cm³
    >>> stats = sample_statistic(densities)
    >>> print_sample_statistics(stats)
    """
    height   = np.atleast_1d(np.array(height,   dtype=float))
    diameter = np.atleast_1d(np.array(diameter, dtype=float))
    mass     = np.atleast_1d(np.array(mass,     dtype=float))

    N = len(height)
    if len(diameter) != N or len(mass) != N:
        raise ValueError(
            f"height, diameter e mass devono avere la stessa lunghezza "
            f"(trovati: {N}, {len(diameter)}, {len(mass)})."
        )

    # rho = 4*m / (pi * d^2 * h)   [g/mm³ se input in g e mm]
    densities = (4.0 * mass) / (np.pi * diameter**2 * height)

    # Conversione g/mm³ → g/cm³: 1 cm³ = 1000 mm³ → moltiplica per 1000
    if to_cm3:
        densities = densities * 1000.0

    # Propagazione incertezze
    has_sigma = (sigma_height is not None or
                sigma_diameter is not None or
                sigma_mass is not None)

    if not has_sigma:
        return densities, None

    def _broadcast(s, name):
        """Converte sigma in array di lunghezza N, controllando la forma."""
        s = np.array(s, dtype=float)
        if s.ndim == 0:
            return np.full(N, float(s))
        if len(s) != N:
            raise ValueError(
                f"sigma_{name} ha lunghezza {len(s)}, attesa {N}."
            )
        return s

    # Usa zero se la sigma non è fornita (contributo nullo alla propagazione)
    sh = _broadcast(sigma_height,   "height")   if sigma_height   is not None else np.zeros(N)
    sd = _broadcast(sigma_diameter, "diameter") if sigma_diameter is not None else np.zeros(N)
    sm = _broadcast(sigma_mass,     "mass")     if sigma_mass     is not None else np.zeros(N)

    # Propagazione relativa: (sigma_rho/rho)^2 = (sm/m)^2 + (2*sd/d)^2 + (sh/h)^2
    # La proporzione relativa è invariante alla conversione di unità
    rel_var = (sm / mass)**2 + (2.0 * sd / diameter)**2 + (sh / height)**2
    sigma_densities = densities * np.sqrt(rel_var)

    return densities, sigma_densities


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
    y : array-like
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
        Coefficiente di correlazione tra m e c.
    chi2 : float
        Chi quadro del fit.
    chi2_rid : float
        Chi quadro ridotto (chi2 / (N-2)).
    N : int
        Numero di punti.
    weighted : bool
        True se il fit è pesato.

    Eccezioni
    ---------
    ValueError
        Se N < 3, sigma_y contiene zeri, o Delta = 0.

    Esempio
    -------
    >>> risultati = linear_fit_complete(x, y, sigma_y)
    >>> print_fit_results(risultati)
    >>> m, c, sig_m, sig_c, cov_mc, *_ = risultati
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

    return m, c, sigma_m, sigma_c, cov_mc, rho, chi2, chi2_rid, N, weighted


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
    y : array-like

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
    >>> fit_res, sigma_emp, residui = linear_fit_residuals(massa, volume)
    >>> print(f"sigma empirica dai residui: {sigma_emp:.6f}")
    >>> print_fit_results(fit_res)
    >>> m, c, sm, sc, cov, *_ = fit_res
    >>> sigma_y_plot = np.full_like(massa, sigma_emp)
    >>> plot_generic_fit(massa, volume, sigma_y_plot, m, c, sm, sc, cov)
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

def print_sample_statistics(results, color=None):
    """
    Stampa le statistiche campionarie.

    Parametri
    ----------
    results : tuple (mean, var, std, std_mean, N)
        Output di `sample_statistic`.
    color : str, opzionale
        Colore dell'intestazione e del separatore ('blue', 'red',
        'green', 'yellow', 'magenta'). Default None (nessun colore).

    Esempio
    -------
    >>> stats = sample_statistic(data)
    >>> print_sample_statistics(stats, color="blue")
    """
    mean, var, std, std_mean, N = results
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
    sigma_offset : float
    info : dict
    unit : str, opzionale
    color : str, opzionale

    Esempio
    -------
    >>> print_offset_results(offset, sigma_offset, info, unit="mm", color="magenta")
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
    sigma_a : float
    sigma_b : float
    sigma_tot : float
    unit : str, opzionale
    color : str, opzionale

    Esempio
    -------
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
    mean : float
    sigma_total : float
    unit : str, opzionale
    color : str, opzionale

    Esempio
    -------
    >>> print_final_result("Nonio", res_nonio['mean'], res_nonio['sigma_total'], "mm", color="blue")
    """
    text = f"RISULTATO FINALE: {name} = {mean:.6f} ± {sigma_total:.6f} {unit}"
    print(_apply_color(text, color))


def print_fit_results(results):
    """
    Stampa i risultati del fit lineare in formato leggibile.

    Parametri
    ----------
    results : tuple
        Output di `linear_fit_complete`.

    Esempio
    -------
    >>> print_fit_results(linear_fit_complete(x, y, sig_y))
    """
    m, c, sigma_m, sigma_c, cov_mc, rho, chi2, chi2_rid, N, weighted = results

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
                     xlabel="X", ylabel="Y", title="Fit Lineare", plot=True, save=True, type=""):
    """
    Visualizza i dati sperimentali con barre di errore e il fit lineare.

    Parametri
    ----------
    x, y : array-like
    sig_y : array-like
    m, c : float
    sig_m, sig_c : float
    cov_mc : float, opzionale
    xlabel, ylabel : str, opzionale

    Esempio
    -------
    >>> plot_generic_fit(x, y, sig_y, m, c, sig_m, sig_c, cov_mc,
    ...                  xlabel="Tempo [s]", ylabel="Spazio [m]")
    """    

    x_range = np.linspace(min(x), max(x), 100)
    y_fit = m * x_range + c
    plt.plot(x_range, y_fit, 'r', label=f'Fit {type} : y = {m:.4f}x + {c:.4f}')

    sigma_band = np.sqrt(np.maximum(0, (x_range * sig_m)**2 + sig_c**2 + 2 * x_range * cov_mc))
    plt.fill_between(x_range, y_fit - sigma_band, y_fit + sigma_band,
                     alpha=0.2, color='r', label=f'Banda incertezza {type}')

    plt.errorbar(x, y, yerr=sig_y, markersize=5, label='Dati Sperimentali')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig("fit_lineare.png", dpi=500)
    if plot:        
        plt.show()


def plot_normal(mean, sigma, label=None, color=None, unit=""):
    """
    Disegna una distribuzione normale sul grafico corrente (axes attivo).

    Parametri
    ----------
    mean : float
    sigma : float
    label : str, opzionale
    color : str, opzionale
    unit : str, opzionale

    Esempio
    -------
    >>> plt.figure(figsize=(10, 6))
    >>> lt.plot_normal(res_nonio['mean'],  res_nonio['sigma_total'],
    ...                label="Nonio",  color="blue",  unit="mm")
    >>> plt.show()
    """
    from scipy.stats import norm as _norm

    x = np.linspace(mean - 1000*sigma, mean + 1000*sigma, 100000)
    y = _norm.pdf(x, mean, sigma)

    if label is None:
        label = f"μ = {mean:.4f} {unit}  σ = {sigma:.4f} {unit}"
    else:
        label = label + ": " + f"μ = {mean:.4f} {unit}  σ = {sigma:.4f} {unit}"

    plt.plot(x, y, color=color, label=label)
    plt.fill_between(x, y, color=color, alpha=0.1)
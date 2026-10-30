import lab_tools as lt

RES_TEMPO = 0.01

x = [
    4.54,
    4.44,
    4.42,
    4.45,
    4.55,
    4.48,
    4.42,
    4.50,
    4.50,
    4.50,
    4.47,
    4.35,
    4.52,
    4.59,
    4.55,
    4.51,
    4.64,
    4.57,
    4.44,
    4.50
]

risultati = lt.sample_statistic(x)

sigma_b = lt.resolution_uncertainty(RES_TEMPO)

final_res = lt.get_total_uncertainty(risultati[0], risultati[3], sigma_b)

lt.print_total_uncertainty(final_res['mean'], final_res['sigma_a'], final_res['sigma_b'], final_res['sigma_total'])
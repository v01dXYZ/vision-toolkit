# -*- coding: utf-8 -*-

import numpy as np


def modified_k_coefficient(scanpath):

    values = np.asarray(scanpath.values)

    # Vérif basique de la forme
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError(
            "modified_k_coefficient requires scanpath.values with at least 3 rows (x, y, duration)."
        )

    n_ = values.shape[1]
    if n_ < 2:
        return np.nan
 
    d = values[2]
    mu_d = np.mean(d)

    std_d = np.std(d, ddof=1)
 
    a_s = np.linalg.norm(values[0:2, 1:] - values[0:2, :-1], axis=0)
    mu_a = np.mean(a_s)
    std_a = np.std(a_s, ddof=1)
 
    if std_d == 0 or std_a == 0 or not np.isfinite(std_d) or not np.isfinite(std_a):
        return np.nan
 
    k_j = (d[:-1] - mu_d) / std_d - (a_s - mu_a) / std_a
 
    denom = float(n_ - 1)
    if denom <= 0:
        return np.nan

    k_c = np.sum(k_j) / denom

    return float(k_c)
# -*- coding: utf-8 -*-

import numpy as np

from vision_toolkit.visualization.scanpath.single.geometrical import plot_BCEA


def bcea(scanpath, probability):
 
    def pearson_corr(x, y):
      
        x = np.asarray(x)
        y = np.asarray(y)

        if x.size != y.size:
            raise ValueError("x and y must have the same length for Pearson correlation.")

        if x.size < 2: 
            return np.nan

        mx = x.mean()
        my = y.mean()

        xm, ym = x - mx, y - my

        _num = np.sum(xm * ym)
        _den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))

        if _den == 0:
            return 0.0

        p_c = _num / _den
        p_c = max(min(p_c, 1.0), -1.0)

        return p_c
 
    if not (0.0 < probability < 1.0):
        raise ValueError("probability must be in (0, 1).")

    values = np.asarray(scanpath.values)
    if values.shape[0] < 2:
        raise ValueError("scanpath.values must have at least 2 rows (x, y).")

    x_a = values[0]
    y_a = values[1]

    if x_a.size < 2:
        raise ValueError("BCEA requires at least 2 samples.")
 
    k = -np.log(1.0 - probability) 
    p_c = pearson_corr(x_a, y_a)
 
    sd_x = np.std(x_a, ddof=1)
    sd_y = np.std(y_a, ddof=1)
 
    if sd_x == 0 or sd_y == 0 or not np.isfinite(sd_x) or not np.isfinite(sd_y):
        bcea_val = 0.0
    else: 
        one_minus_r2 = 1.0 - p_c**2
        if one_minus_r2 < 0: 
            one_minus_r2 = max(one_minus_r2, 0.0)

        bcea_val = 2.0 * np.pi * k * sd_x * sd_y * np.sqrt(one_minus_r2)
 
    if scanpath.config.get("display_results", False):
        plot_BCEA(scanpath.values, probability, scanpath.config.get("display_path"), scanpath.config)

    return bcea_val


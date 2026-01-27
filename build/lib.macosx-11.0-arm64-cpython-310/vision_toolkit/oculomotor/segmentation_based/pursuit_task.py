
import numpy as np
import pandas as pd

import scipy
import scipy.optimize as optimize

from vision_toolkit.segmentation.processing.ternary_segmentation import TernarySegmentation
from vision_toolkit.utils.segmentation_utils import interval_merging


class PursuitTask(TernarySegmentation):
    """
    Smooth pursuit task analysis aligned on the theoretical pursuit window.

    Assumptions
    -----------
    - pursuit_start_idx indicates when the subject starts tracking the target (viewer timeline)
    - No temporal lag estimation is performed
    - The theoretical trajectory defines the analysis window
    - Viewer data are sliced directly on that window
    - pursuit_intervals are expressed in the *window* coordinates [0 .. nb_samples_pursuit-1]
      and are inclusive: [start, end]
    """

    def __init__(self, input, theoretical_df, **kwargs):
        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing Pursuit Task...\n")

        # Reuse an already computed segmentation
        if isinstance(input, TernarySegmentation):
            self.__dict__ = input.__dict__.copy()
            self.config.update({"verbose": verbose})
        else:
            sampling_frequency = kwargs.get("sampling_frequency", None)
            assert sampling_frequency is not None, "Sampling frequency must be specified"
            super().__init__(input, **kwargs)

        self.s_f = float(self.config["sampling_frequency"])
 
        if isinstance(theoretical_df, pd.DataFrame):
            t_df = theoretical_df
        else:
            t_df = pd.read_csv(theoretical_df)

        x_theo = np.asarray(t_df.iloc[:, 0], dtype=np.float64)
        y_theo = np.asarray(t_df.iloc[:, 1], dtype=np.float64)
        n_theo = int(len(x_theo))
 
        start_idx = int(self.config.get("pursuit_start_idx", 0))
        start_idx = max(start_idx, 0)

        nb_samples = int(self.config.get("nb_samples", len(self.data_set["x_array"])))
        end_idx = min(start_idx + n_theo, nb_samples)

        # Truncate theoretical to match available viewer window length
        n_win = max(0, end_idx - start_idx)
        x_theo = x_theo[:n_win]
        y_theo = y_theo[:n_win]

        self.config["nb_samples_pursuit"] = int(n_win)
        self.config["pursuit_start_idx"] = int(start_idx)
        self.config["pursuit_end_idx"] = int(end_idx)

        # Build aligned window arrays (NO lag estimation)
        self._align_on_theoretical_window(x_theo=x_theo, y_theo=y_theo)

        # Reproject pursuit intervals into window coordinates [0..n_win-1]
        self.pursuit_intervals = self._reproject_intervals_to_window(
            self.segmentation_results.get("pursuit_intervals", []),
            start_idx,
            end_idx,
        )

        if verbose:
            print("...Pursuit Task done\n")
 
    
    def _align_on_theoretical_window(self, x_theo, y_theo):
        """
        Align viewer data with the theoretical pursuit window (no lag estimation).

        Creates:
            - x_pursuit, y_pursuit: viewer positions in the window
            - x_theo_pursuit, y_theo_pursuit: theoretical positions (same length)
        """
        start_idx = int(self.config["pursuit_start_idx"])
        n = int(self.config["nb_samples_pursuit"])

        x_view = np.asarray(self.data_set["x_array"], dtype=np.float64)
        y_view = np.asarray(self.data_set["y_array"], dtype=np.float64)

        end_idx = min(start_idx + n, len(x_view), len(y_view))
        n_valid = end_idx - start_idx

        if n_valid <= 0:
            x_win = np.array([], dtype=np.float64)
            y_win = np.array([], dtype=np.float64)
            x_theo_win = np.array([], dtype=np.float64)
            y_theo_win = np.array([], dtype=np.float64)
        else:
            x_win = x_view[start_idx:end_idx]
            y_win = y_view[start_idx:end_idx]
            x_theo_win = np.asarray(x_theo[:n_valid], dtype=np.float64)
            y_theo_win = np.asarray(y_theo[:n_valid], dtype=np.float64)

        self.data_set.update(
            {
                "x_pursuit": x_win,
                "y_pursuit": y_win,
                "x_theo_pursuit": x_theo_win,
                "y_theo_pursuit": y_theo_win,
            }
        )

        # Update actual end index in case of truncation
        self.config["pursuit_end_idx"] = int(start_idx + len(x_win))
        self.config["nb_samples_pursuit"] = int(len(x_win))


    def _reproject_intervals_to_window(self, intervals, start_idx, end_idx):
        """
        Convert global pursuit intervals (inclusive) into local window coordinates (inclusive).

        Global interval [a, b] -> intersection with [start_idx, end_idx-1], then shift by -start_idx:
            [max(a, start_idx) - start_idx,
             min(b, end_idx - 1) - start_idx]
        """
        local_intervals = []

        for a, b in intervals:
            a = int(a)
            b = int(b)

            aa = max(a, start_idx)
            bb = min(b, end_idx - 1)

            if bb >= aa:
                local_intervals.append([aa - start_idx, bb - start_idx])

        if len(local_intervals) > 1:
            local_intervals = interval_merging(np.asarray(local_intervals, dtype=int)).tolist()

        return local_intervals


    def _safe_sd(self, x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.nanstd(x, ddof=1)) if np.sum(np.isfinite(x)) >= 2 else 0.0


    def _speed_array_window(self):
        """
        Return absolute_speed restricted to the theoretical window (global slice).

        Note: absolute_speed is defined on transitions; we keep a safe slice and later
        slice per-interval as start..end (exclusive end) in window coordinates.
        """
        a_sp = np.asarray(self.data_set["absolute_speed"], dtype=np.float64)
        s = max(int(self.config["pursuit_start_idx"]), 0)
        e = min(int(self.config["pursuit_end_idx"]), a_sp.size)
        
        return a_sp[s:e]


    def _speed_segment_window(self, start, end):
        """
        Return a speed segment in window coordinates.

        Conventions:
        - Intervals are inclusive on positions [start, end]
        - Speeds are defined on transitions, so use [start, end) where end is inclusive+1
        """
        if end <= start:
            return np.array([], dtype=np.float64)

        a_sp = self._speed_array_window()
        start = max(int(start), 0)
        end = min(int(end), a_sp.size)
        
        return a_sp[start:end]

  
    def pursuit_task_count(self):
        
        ct = int(len(self.pursuit_intervals))
        return {"count": ct}


    def pursuit_task_frequency(self):
        ct = float(len(self.pursuit_intervals))
        denom = float(self.config["nb_samples_pursuit"]) / float(self.config["sampling_frequency"])
        f = ct / denom if denom > 0 else np.nan
        
        return {"frequency": float(f)}


    def pursuit_task_durations(self, get_raw=True):
        """
        Duration per interval (seconds). Intervals are inclusive in window coords.
        """
        if len(self.pursuit_intervals) == 0:
            res = {"duration mean": np.nan, "duration sd": 0.0, "raw": np.array([], dtype=np.float64)}
            if not get_raw:
                del res["raw"]
            return res

        a_i = np.asarray(self.pursuit_intervals, dtype=np.int64)
        a_d = (a_i[:, 1] - a_i[:, 0] + 1) / self.s_f

        res = {
            "duration mean": float(np.nanmean(a_d)),
            "duration sd": self._safe_sd(a_d),
            "raw": a_d,
        }
        if not get_raw:
            del res["raw"]
            
        return res


    def pursuit_task_proportion(self):
        """
        Proportion of time spent in pursuit intervals within the theoretical window.
        """
        if not self.pursuit_intervals:
            return {"task proportion": 0.0}

        intervals = np.asarray(self.pursuit_intervals, dtype=np.int64)
        valid = intervals[intervals[:, 1] >= intervals[:, 0]]
        if valid.size == 0:
            return {"task proportion": 0.0}

        total_pursuit_samples = float(np.sum(valid[:, 1] - valid[:, 0] + 1))
        total_samples = float(self.config.get("nb_samples_pursuit", 0))

        if total_samples <= 0:
            return {"task proportion": 0.0}

        return {"task proportion": float(total_pursuit_samples / total_samples)}


    def pursuit_task_velocity(self, get_raw=True):
        """
        Pooled speeds across all pursuit segments in the window.
        For an interval [a,b] (inclusive in positions), use speeds [a, b) in window coords,
        i.e. end = b (inclusive) -> b is used as exclusive end.
        """
        l_sp = []
        for a, b in self.pursuit_intervals:
            seg = self._speed_segment_window(a, b)  # [a, b)
            if seg.size:
                l_sp.append(seg)

        all_sp = np.concatenate(l_sp) if len(l_sp) else np.array([], dtype=np.float64)

        res = {
            "velocity mean": float(np.nanmean(all_sp)) if all_sp.size else np.nan,
            "velocity sd": self._safe_sd(all_sp) if all_sp.size else np.nan,
            "raw": all_sp,
        }
        if not get_raw:
            del res["raw"]
            
        return res


    def pursuit_task_velocity_means(self, get_raw=True):
        """
        Mean speed per pursuit interval.
        """
        means = []
        for a, b in self.pursuit_intervals:
            seg = self._speed_segment_window(a, b)
            means.append(float(np.nanmean(seg)) if seg.size else np.nan)

        means = np.asarray(means, dtype=np.float64)

        res = {
            "velocity mean mean": float(np.nanmean(means)) if means.size else np.nan,
            "velocity mean sd": self._safe_sd(means) if means.size else np.nan,
            "raw": means,
        }
        if not get_raw:
            del res["raw"]
            
        return res


    def pursuit_task_peak_velocity(self, get_raw=True):
        """
        Peak speed per pursuit interval.
        """
        peaks = []
        for a, b in self.pursuit_intervals:
            seg = self._speed_segment_window(a, b)
            peaks.append(float(np.nanmax(seg)) if seg.size else np.nan)

        peaks = np.asarray(peaks, dtype=np.float64)

        res = {
            "velocity peak mean": float(np.nanmean(peaks)) if peaks.size else np.nan,
            "velocity peak sd": self._safe_sd(peaks) if peaks.size else np.nan,
            "raw": peaks,
        }
        if not get_raw:
            del res["raw"]
            
        return res


    def pursuit_task_amplitude(self, get_raw=True):
        """
        Straight-line amplitude (start->end) per pursuit interval, using global arrays but
        indices converted to global by adding pursuit_start_idx.
        """
        x_a = np.asarray(self.data_set["x_array"], dtype=np.float64)
        y_a = np.asarray(self.data_set["y_array"], dtype=np.float64)
        z_a = np.asarray(self.data_set["z_array"], dtype=np.float64)

        dist_ = self.distances[self.config["distance_type"]]
        s0 = int(self.config["pursuit_start_idx"])

        dsp = []
        for a, b in self.pursuit_intervals:
            ga = s0 + int(a)
            gb = s0 + int(b)

            if gb < ga or gb >= len(x_a) or ga < 0:
                dsp.append(np.nan)
                continue

            p0 = np.array([x_a[ga], y_a[ga], z_a[ga]])
            p1 = np.array([x_a[gb], y_a[gb], z_a[gb]])
            dsp.append(dist_(p0, p1))

        dsp = np.asarray(dsp, dtype=np.float64)

        res = {
            "pursuit amplitude mean": float(np.nanmean(dsp)) if dsp.size else np.nan,
            "pursuit amplitude sd": self._safe_sd(dsp) if dsp.size else np.nan,
            "raw": dsp,
        }
        if not get_raw:
            del res["raw"]
            
        return res


    def pursuit_task_distance(self, get_raw=True):
        """
        Cumulative distance along the path per pursuit interval.
        """
        x_a = np.asarray(self.data_set["x_array"], dtype=np.float64)
        y_a = np.asarray(self.data_set["y_array"], dtype=np.float64)
        z_a = np.asarray(self.data_set["z_array"], dtype=np.float64)

        dist_ = self.distances[self.config["distance_type"]]
        s0 = int(self.config["pursuit_start_idx"])

        t_cum = []
        for a, b in self.pursuit_intervals:
            ga = s0 + int(a)
            gb = s0 + int(b)

            if gb <= ga or ga < 0 or gb >= len(x_a):
                t_cum.append(np.nan)
                continue

            l_cum = 0.0
            for k in range(ga, gb):
                p0 = np.array([x_a[k], y_a[k], z_a[k]])
                p1 = np.array([x_a[k + 1], y_a[k + 1], z_a[k + 1]])
                l_cum += dist_(p0, p1)

            t_cum.append(l_cum)

        t_cum = np.asarray(t_cum, dtype=np.float64)

        res = {
            "pursuit cumul. distance mean": float(np.nanmean(t_cum)) if t_cum.size else np.nan,
            "pursuit cumul. distance sd": self._safe_sd(t_cum) if t_cum.size else np.nan,
            "raw": t_cum,
        }
        if not get_raw:
            del res["raw"]
            
        return res


    def pursuit_task_efficiency(self, get_raw=True):
        """
        Efficiency = amplitude / cumulative distance per interval.
        """
        amp = self.pursuit_task_amplitude(get_raw=True)["raw"]
        dist = self.pursuit_task_distance(get_raw=True)["raw"]

        if amp.size == 0 or dist.size == 0:
            res = {"pursuit efficiency mean": np.nan, "pursuit efficiency sd": np.nan, "raw": np.array([], dtype=np.float64)}
            if not get_raw:
                del res["raw"]
            return res

        eff = np.where(np.isfinite(dist) & (dist > 0), amp / dist, np.nan)
        eff = np.asarray(eff, dtype=np.float64)

        res = {
            "pursuit efficiency mean": float(np.nanmean(eff)) if eff.size else np.nan,
            "pursuit efficiency sd": self._safe_sd(eff) if eff.size else np.nan,
            "raw": eff,
        }
        if not get_raw:
            del res["raw"]
            
        return res


    def pursuit_task_slope_ratios(self, get_raw=True):
        """
        Per-interval slope ratios (eye slope / target slope) for x and y.

        Important: returns one value per interval (np.nan if invalid) to keep
        alignment with durations/weights.
        """
        _ints = self.pursuit_intervals
        d_t = 1.0 / float(self.config["sampling_frequency"])
        s_idx = int(self.config["pursuit_start_idx"])

        pos = {"x": self.data_set["x_pursuit"], "y": self.data_set["y_pursuit"]}
        theo = {"x": self.data_set["x_theo_pursuit"], "y": self.data_set["y_theo_pursuit"]}

        s_r = {"x": [], "y": []}

        for a, b in _ints:
            # Map window interval [a,b] to theoretical indices [a,b] (same window length),
            # but keep bounds safe.
            theo_start = max(0, int(a))
            theo_end = min(len(theo["x"]), int(b) + 1)

            for _dir in ["x", "y"]:
                l_p_e = np.asarray(pos[_dir][int(a) : int(b) + 1], dtype=np.float64)
                l_p_t = np.asarray(theo[_dir][theo_start:theo_end], dtype=np.float64)

                min_len = min(len(l_p_e), len(l_p_t))
                if min_len < 2:
                    s_r[_dir].append(np.nan)
                    continue

                l_p_e = l_p_e[:min_len]
                l_p_t = l_p_t[:min_len]
                l_x = np.arange(min_len, dtype=np.float64) * d_t

                try:
                    slope_e = np.polyfit(l_x, l_p_e, deg=1)[0]
                    slope_t = np.polyfit(l_x, l_p_t, deg=1)[0]
                    s_r[_dir].append(slope_e / slope_t if slope_t != 0 else np.nan)
                except Exception:
                    s_r[_dir].append(np.nan)

        for _dir in ["x", "y"]:
            s_r[_dir] = np.asarray(s_r[_dir], dtype=np.float64)

        res = {"slope ratios": s_r}
        if not get_raw:
            # keep API symmetry with others
            pass
        
        return res


    def pursuit_task_slope_gain(self, _type="weighted"):
        """
        Aggregate slope ratios into a gain (x and y).

        Parameters
        ----------
        _type : str
            'mean' or 'weighted' (duration-weighted mean)

        Returns
        -------
        dict
            {'slope gain': {'gain x': float, 'gain y': float}}
        """
        if not self.pursuit_intervals:
            return {"slope gain": {"gain x": 0.0, "gain y": 0.0}}

        slope_ratios = self.pursuit_task_slope_ratios(get_raw=True)["slope ratios"]

        intervals = np.asarray(self.pursuit_intervals, dtype=np.int64)
        durations = (intervals[:, 1] - intervals[:, 0] + 1).astype(np.float64)
        valid_dur = durations > 0

        gains = {}
        for direction in ["x", "y"]:
            ratios = np.asarray(slope_ratios.get(direction, np.array([], dtype=np.float64)), dtype=np.float64)

            if ratios.size != len(self.pursuit_intervals):
                gains["gain " + direction] = 0.0
                continue

            valid = valid_dur & np.isfinite(ratios)
            if not np.any(valid):
                gains["gain " + direction] = 0.0
                continue

            if _type == "mean":
                gains["gain " + direction] = float(np.nanmean(ratios[valid]))
            else:  # weighted
                w = durations[valid]
                r = ratios[valid]
                denom = float(np.sum(w))
                gains["gain " + direction] = float(np.sum(w * r) / denom) if denom > 0 else 0.0

        return {"slope gain": gains}


    def pursuit_task_crossing_time(self, tolerance=1.0):
        """
        First time (seconds) the eye position matches the theoretical target position
        within a tolerance, computed on the window arrays.
        """
        s_f = float(self.config["sampling_frequency"])
        n = int(self.config.get("nb_samples_pursuit", len(self.data_set["x_pursuit"])))
        time = np.arange(n, dtype=np.float64) / s_f

        crossings = {"x": None, "y": None}

        for direction in ["x", "y"]:
            eye_pos = np.asarray(self.data_set[f"{direction}_pursuit"], dtype=np.float64)
            target_pos = np.asarray(self.data_set[f"{direction}_theo_pursuit"], dtype=np.float64)

            min_len = min(len(eye_pos), len(target_pos), len(time))
            if min_len <= 0:
                continue

            eye_pos = eye_pos[:min_len]
            target_pos = target_pos[:min_len]
            tt = time[:min_len]

            error = np.abs(eye_pos - target_pos)
            idx = np.where(error < float(tolerance))[0]
            if idx.size > 0:
                crossings[direction] = float(tt[idx[0]])

        return {"crossing time": crossings}


    def pursuit_task_overall_gain(self, get_raw=True):
        """
        Gain based on speed magnitude: ||v_eye|| / ||v_target|| pooled over intervals.
        Computed on the window arrays.
        """
        d_t = 1.0 / float(self.config["sampling_frequency"])

        pos = {"x": np.asarray(self.data_set["x_pursuit"], dtype=np.float64),
               "y": np.asarray(self.data_set["y_pursuit"], dtype=np.float64)}
        theo = {"x": np.asarray(self.data_set["x_theo_pursuit"], dtype=np.float64),
                "y": np.asarray(self.data_set["y_theo_pursuit"], dtype=np.float64)}

        gains = []

        for a, b in self.pursuit_intervals:
            a = int(a); b = int(b)
            if b <= a:
                continue

            xe = pos["x"][a:b+1]
            ye = pos["y"][a:b+1]
            xt = theo["x"][a:b+1]
            yt = theo["y"][a:b+1]

            min_len = min(len(xe), len(ye), len(xt), len(yt))
            if min_len < 2:
                continue

            xe = xe[:min_len]; ye = ye[:min_len]
            xt = xt[:min_len]; yt = yt[:min_len]

            se_x = xe[1:] - xe[:-1]
            se_y = ye[1:] - ye[:-1]
            st_x = xt[1:] - xt[:-1]
            st_y = yt[1:] - yt[:-1]

            e_vel = np.sqrt((se_x / d_t) ** 2 + (se_y / d_t) ** 2)
            t_vel = np.sqrt((st_x / d_t) ** 2 + (st_y / d_t) ** 2)

            with np.errstate(divide="ignore", invalid="ignore"):
                l_g = e_vel / t_vel
            gains += list(l_g[np.isfinite(l_g)])

        gains = np.asarray(gains, dtype=np.float64)

        res = {"overall gain": float(np.nanmean(gains)) if gains.size else np.nan,
               "raw": gains}
        if not get_raw:
            del res["raw"]
            
        return res


    def pursuit_task_overall_gain_x(self, get_raw=True):
        """
        Gain based on x-velocity: v_eye_x / v_target_x pooled over intervals.
        """
        d_t = 1.0 / float(self.config["sampling_frequency"])

        xe_all = np.asarray(self.data_set["x_pursuit"], dtype=np.float64)
        xt_all = np.asarray(self.data_set["x_theo_pursuit"], dtype=np.float64)

        gains = []

        for a, b in self.pursuit_intervals:
            a = int(a); b = int(b)
            if b <= a:
                continue

            xe = xe_all[a:b+1]
            xt = xt_all[a:b+1]

            min_len = min(len(xe), len(xt))
            if min_len < 2:
                continue

            xe = xe[:min_len]
            xt = xt[:min_len]

            se = xe[1:] - xe[:-1]
            st = xt[1:] - xt[:-1]

            e_vel = se / d_t
            t_vel = st / d_t

            with np.errstate(divide="ignore", invalid="ignore"):
                l_g = e_vel / t_vel
            gains += list(l_g[np.isfinite(l_g)])

        gains = np.asarray(gains, dtype=np.float64)

        res = {"overall gain x": float(np.nanmean(gains)) if gains.size else np.nan,
               "raw": gains}
        if not get_raw:
            del res["raw"]
            
        return res


    def pursuit_task_overall_gain_y(self, get_raw=True):
        """
        Gain based on y-velocity: v_eye_y / v_target_y pooled over intervals.
        """
        d_t = 1.0 / float(self.config["sampling_frequency"])

        ye_all = np.asarray(self.data_set["y_pursuit"], dtype=np.float64)
        yt_all = np.asarray(self.data_set["y_theo_pursuit"], dtype=np.float64)

        gains = []

        for a, b in self.pursuit_intervals:
            a = int(a); b = int(b)
            if b <= a:
                continue

            ye = ye_all[a:b+1]
            yt = yt_all[a:b+1]

            min_len = min(len(ye), len(yt))
            if min_len < 2:
                continue

            ye = ye[:min_len]
            yt = yt[:min_len]

            se = ye[1:] - ye[:-1]
            st = yt[1:] - yt[:-1]

            e_vel = se / d_t
            t_vel = st / d_t

            with np.errstate(divide="ignore", invalid="ignore"):
                l_g = e_vel / t_vel
            gains += list(l_g[np.isfinite(l_g)])

        gains = np.asarray(gains, dtype=np.float64)

        res = {"overall gain y": float(np.nanmean(gains)) if gains.size else np.nan,
               "raw": gains}
        if not get_raw:
            del res["raw"]
            
        return res


    def pursuit_task_sinusoidal_phase(self):
        """
        Fit a sinusoid on theoretical x and viewer x (window), and return phase difference.
        Output kept identical: {'phase difference': float}
        """

        def fit_sin(tt, yy):
            """
            Fit a sinusoid to a time series.
            Returns dict with phase and a callable fitfunc.
            """
            tt = np.asarray(tt, dtype=np.float64)
            yy = np.asarray(yy, dtype=np.float64)

            if tt.size < 3 or yy.size < 3:
                raise ValueError("Not enough samples for sinusoidal fit.")

            ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
            Fyy = np.abs(np.fft.fft(yy))
            guess_freq = np.abs(ff[np.argmax(Fyy[1:]) + 1]) if len(Fyy) > 1 else 1.0
            guess_amp = np.std(yy) * np.sqrt(2.0)
            guess_offset = np.mean(yy)
            guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset], dtype=np.float64)

            def sinfunc(t, A, w, p, c):
                return A * np.sin(w * t + p) + c

            popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess, maxfev=20000)
            A, w, p, c = popt
            f = w / (2.0 * np.pi)
            fitfunc = lambda t: A * np.sin(w * t + p) + c

            return {
                "amp": A,
                "omega": w,
                "phase": p,
                "offset": c,
                "freq": f,
                "period": 1.0 / f if f != 0 else np.inf,
                "fitfunc": fitfunc,
                "maxcov": float(np.max(pcov)) if pcov.size else np.nan,
                "rawres": (guess, popt, pcov),
            }

        s_f = float(self.config["sampling_frequency"])
        d_t = 1.0 / s_f

        x_t = np.asarray(self.data_set["x_theo_pursuit"], dtype=np.float64)
        x_e = np.asarray(self.data_set["x_pursuit"], dtype=np.float64)

        n = min(x_t.size, x_e.size)
        if n < 3:
            return {"phase difference": np.nan}

        x_t = x_t[:n]
        x_e = x_e[:n]

        tt = np.linspace(0.0, (n - 1) * d_t, n, dtype=np.float64)

        try:
            res1 = fit_sin(tt, x_t)
            res2 = fit_sin(tt, x_e)
        except Exception:
            return {"phase difference": np.nan}
 
        return {"phase difference": float(res1["phase"] - res2["phase"])}


    def pursuit_task_accuracy(self, pursuit_accuracy_tolerance=0.15, _type="mean"):
        """
        Accuracy based on slope ratios closeness to 1 within tolerance.

        Outputs kept identical to original code:
            returns dict({'x': value, 'y': value})
        """
        if not self.pursuit_intervals:
            return {"x": 0.0, "y": 0.0}

        intervals = np.asarray(self.pursuit_intervals, dtype=np.int64)
        durations = (intervals[:, 1] - intervals[:, 0] + 1).astype(np.float64)
        valid_dur = durations > 0

        s_r = self.pursuit_task_slope_ratios(get_raw=True)["slope ratios"]
        ac_t = float(pursuit_accuracy_tolerance)

        out = {}
        for _dir in ["x", "y"]:
            ratios = np.asarray(s_r.get(_dir, np.array([], dtype=np.float64)), dtype=np.float64)
            if ratios.size != len(self.pursuit_intervals):
                out[_dir] = 0.0
                continue

            # within [1-ac_t, 1+ac_t]
            within = np.where((ratios < 1.0 + ac_t) & (ratios > 1.0 - ac_t), 1.0, 0.0)
            within = np.where(np.isfinite(ratios), within, 0.0)

            if _type == "weighted":
                mask = valid_dur & np.isfinite(ratios)
                denom = float(np.sum(durations[mask])) if np.any(mask) else 0.0
                out[_dir] = float(np.sum(within[mask] * durations[mask]) / denom) if denom > 0 else 0.0
            else:  # 'mean'
                out[_dir] = float(np.mean(within)) if within.size else 0.0

        return {'accuracy': out}


    def ap_entropy(self, diff_vec, w_s, t_eps):
        """
        Approximate entropy helper (kept identical in spirit).
        """
        diff_vec = np.asarray(diff_vec, dtype=np.float64)
        n_s = len(diff_vec)
        w_s = int(w_s)
        t_eps = float(t_eps)

        if n_s <= w_s + 1 or w_s < 1:
            return np.nan

        x_m = np.zeros((n_s - w_s + 1, w_s), dtype=np.float64)
        x_mp = np.zeros((n_s - w_s, w_s + 1), dtype=np.float64)

        for i in range(n_s - w_s + 1):
            x_m[i] = diff_vec[i : i + w_s]
            if i < n_s - w_s:
                x_mp[i] = diff_vec[i : i + w_s + 1]

        C_m = np.zeros(n_s - w_s + 1, dtype=np.float64)
        C_mp = np.zeros(n_s - w_s, dtype=np.float64)

        for i in range(n_s - w_s + 1):
            d = np.abs(x_m - x_m[i])
            d_m = np.sum(np.max(d, axis=1) < t_eps)
            C_m[i] = d_m / (n_s - w_s + 1)

        for i in range(n_s - w_s):
            d = np.abs(x_mp - x_mp[i])
            d_mp = np.sum(np.max(d, axis=1) < t_eps)
            C_mp[i] = d_mp / (n_s - w_s)

        # Avoid log(0)
        C_m = np.clip(C_m, 1e-12, None)
        C_mp = np.clip(C_mp, 1e-12, None)

        entropy = np.sum(np.log(C_m)) / len(C_m) - np.sum(np.log(C_mp)) / len(C_mp)
        
        return float(entropy)


    def pursuit_task_entropy(self, pursuit_entropy_window, pursuit_entropy_tolerance):
        """
        Entropy on the speed error (viewer speed - theoretical speed) for x and y.

        Output kept identical to original code:
            returns dict({'x': apEn_x, 'y': apEn_y})
        """
        w_s = int(pursuit_entropy_window)
        t_eps = float(pursuit_entropy_tolerance)

        s_f = float(self.config["sampling_frequency"])

        x_e = np.asarray(self.data_set["x_pursuit"], dtype=np.float64)
        y_e = np.asarray(self.data_set["y_pursuit"], dtype=np.float64)
        x_t = np.asarray(self.data_set["x_theo_pursuit"], dtype=np.float64)
        y_t = np.asarray(self.data_set["y_theo_pursuit"], dtype=np.float64)

        n = min(x_e.size, y_e.size, x_t.size, y_t.size)
        if n < 3:
            return {"x": np.nan, "y": np.nan}

        x_e = x_e[:n]
        y_e = y_e[:n]
        x_t = x_t[:n]
        y_t = y_t[:n]

        # velocities (simple finite difference) in units per second
        sp_e_x = np.zeros(n, dtype=np.float64)
        sp_e_y = np.zeros(n, dtype=np.float64)
        sp_t_x = np.zeros(n, dtype=np.float64)
        sp_t_y = np.zeros(n, dtype=np.float64)

        sp_e_x[:-1] = (x_e[1:] - x_e[:-1]) * s_f
        sp_e_y[:-1] = (y_e[1:] - y_e[:-1]) * s_f
        sp_t_x[:-1] = (x_t[1:] - x_t[:-1]) * s_f
        sp_t_y[:-1] = (y_t[1:] - y_t[:-1]) * s_f

        diff_x = sp_e_x - sp_t_x
        diff_y = sp_e_y - sp_t_y

        return {'entropy':
            {"x": self.ap_entropy(diff_x, w_s, t_eps),
            "y": self.ap_entropy(diff_y, w_s, t_eps),}
        }




def pursuit_task_count(input, theoretical_df, **kwargs):
    if isinstance(input, PursuitTask):
        results = input.pursuit_task_count()
        input.verbose()
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_count()
        pursuit_task.verbose()
    return results


def pursuit_task_frequency(input, theoretical_df, **kwargs):
    if isinstance(input, PursuitTask):
        results = input.pursuit_task_frequency()
        input.verbose()
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_frequency()
        pursuit_task.verbose()
    return results


def pursuit_task_durations(input, theoretical_df, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_durations(get_raw=get_raw)
        input.verbose({"get_raw": get_raw})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_durations(get_raw=get_raw)
        pursuit_task.verbose({"get_raw": get_raw})
    return results


def pursuit_task_proportion(input, theoretical_df, **kwargs):
    if isinstance(input, PursuitTask):
        results = input.pursuit_task_proportion()
        input.verbose()
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_proportion()
        pursuit_task.verbose()
    return results


def pursuit_task_velocity(input, theoretical_df, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_velocity(get_raw=get_raw)
        input.verbose({"get_raw": get_raw})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_velocity(get_raw=get_raw)
        pursuit_task.verbose({"get_raw": get_raw})
    return results


def pursuit_task_velocity_means(input, theoretical_df, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_velocity_means(get_raw=get_raw)
        input.verbose({"get_raw": get_raw})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_velocity_means(get_raw=get_raw)
        pursuit_task.verbose({"get_raw": get_raw})
    return results


def pursuit_task_peak_velocity(input, theoretical_df, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_peak_velocity(get_raw=get_raw)
        input.verbose({"get_raw": get_raw})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_peak_velocity(get_raw=get_raw)
        pursuit_task.verbose({"get_raw": get_raw})
    return results


def pursuit_task_amplitude(input, theoretical_df, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_amplitude(get_raw=get_raw)
        input.verbose({"get_raw": get_raw})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_amplitude(get_raw=get_raw)
        pursuit_task.verbose({"get_raw": get_raw})
    return results


def pursuit_task_distance(input, theoretical_df, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_distance(get_raw=get_raw)
        input.verbose({"get_raw": get_raw})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_distance(get_raw=get_raw)
        pursuit_task.verbose({"get_raw": get_raw})
    return results


def pursuit_task_efficiency(input, theoretical_df, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_efficiency(get_raw=get_raw)
        input.verbose({"get_raw": get_raw})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_efficiency(get_raw=get_raw)
        pursuit_task.verbose({"get_raw": get_raw})
    return results


def pursuit_task_slope_ratios(input, theoretical_df, **kwargs):
    # keep get_raw for symmetry, even if ratios are always returned
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_slope_ratios(get_raw=get_raw)
        input.verbose({"get_raw": get_raw})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_slope_ratios(get_raw=get_raw)
        pursuit_task.verbose({"get_raw": get_raw})
    return results


def pursuit_task_slope_gain(input, theoretical_df, **kwargs):
    _type = kwargs.get("_type", "weighted")

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_slope_gain(_type=_type)
        input.verbose({"_type": _type})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_slope_gain(_type=_type)
        pursuit_task.verbose({"_type": _type})
    return results


def pursuit_task_crossing_time(input, theoretical_df, **kwargs):
    tolerance = kwargs.get("tolerance", 1.0)

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_crossing_time(tolerance=tolerance)
        input.verbose({"tolerance": tolerance})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_crossing_time(tolerance=tolerance)
        pursuit_task.verbose({"tolerance": tolerance})
    return results


def pursuit_task_overall_gain(input, theoretical_df, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_overall_gain(get_raw=get_raw)
        input.verbose({"get_raw": get_raw})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_overall_gain(get_raw=get_raw)
        pursuit_task.verbose({"get_raw": get_raw})
    return results


def pursuit_task_overall_gain_x(input, theoretical_df, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_overall_gain_x(get_raw=get_raw)
        input.verbose({"get_raw": get_raw})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_overall_gain_x(get_raw=get_raw)
        pursuit_task.verbose({"get_raw": get_raw})
    return results


def pursuit_task_overall_gain_y(input, theoretical_df, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_overall_gain_y(get_raw=get_raw)
        input.verbose({"get_raw": get_raw})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_overall_gain_y(get_raw=get_raw)
        pursuit_task.verbose({"get_raw": get_raw})
    return results


def pursuit_task_sinusoidal_phase(input, theoretical_df, **kwargs):
    if isinstance(input, PursuitTask):
        results = input.pursuit_task_sinusoidal_phase()
        input.verbose()
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_sinusoidal_phase()
        pursuit_task.verbose()
    return results


def pursuit_task_accuracy(input, theoretical_df, **kwargs):
    pursuit_accuracy_tolerance = kwargs.get("pursuit_accuracy_tolerance", 0.15)
    _type = kwargs.get("_type", "mean")

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_accuracy(
            pursuit_accuracy_tolerance=pursuit_accuracy_tolerance,
            _type=_type,
        )
        input.verbose({"pursuit_accuracy_tolerance": pursuit_accuracy_tolerance, "_type": _type})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_accuracy(
            pursuit_accuracy_tolerance=pursuit_accuracy_tolerance,
            _type=_type,
        )
        pursuit_task.verbose({"pursuit_accuracy_tolerance": pursuit_accuracy_tolerance, "_type": _type})
    return results


def pursuit_task_entropy(input, theoretical_df, **kwargs):
    pursuit_entropy_window = kwargs.get("pursuit_entropy_window", 10)
    pursuit_entropy_tolerance = kwargs.get("pursuit_entropy_tolerance", 0.1)

    if isinstance(input, PursuitTask):
        results = input.pursuit_task_entropy(
            pursuit_entropy_window=pursuit_entropy_window,
            pursuit_entropy_tolerance=pursuit_entropy_tolerance,
        )
        input.verbose({"pursuit_entropy_window": pursuit_entropy_window, "pursuit_entropy_tolerance": pursuit_entropy_tolerance})
    else:
        pursuit_task = PursuitTask(input, theoretical_df, **kwargs)
        results = pursuit_task.pursuit_task_entropy(
            pursuit_entropy_window=pursuit_entropy_window,
            pursuit_entropy_tolerance=pursuit_entropy_tolerance,
        )
        pursuit_task.verbose({"pursuit_entropy_window": pursuit_entropy_window, "pursuit_entropy_tolerance": pursuit_entropy_tolerance})
    return results









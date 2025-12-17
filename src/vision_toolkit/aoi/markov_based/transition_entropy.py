# -*- coding: utf-8 -*-

import numpy as np

 
class TransitionEntropyAnalysis:
    
    def __init__(self, transition_matrix, tol=1e-12, max_iter=100_000, check=True):
        
        
        self.transition_matrix = np.asarray(transition_matrix, dtype=float)
        if check:
            self._validate_and_normalize_transition_matrix()

        self.state_number = self.transition_matrix.shape[0]
        self.stationary_distribution = self.compute_stationary_distribution(tol=tol, max_iter=max_iter)

        t_m_i   = self.t_mutual_information()
        t_m_i_r = self.t_mutual_information_row()
        t_j_e   = self.t_joint_entropy()
        t_c_e   = self.t_conditional_entropy()
        t_c_e_r = self.t_conditional_entropy_row()
        t_s_e   = self.t_stationary_entropy()

        self.results = {
            "AoI_transition_stationary_entropy": t_s_e,
            "AoI_transition_joint_entropy": t_j_e,
            "AoI_transition_conditional_entropy": t_c_e,
            "AoI_transition_conditional_entropy_row": t_c_e_r,
            "AoI_transition_mutual_information": t_m_i,
            "AoI_transition_mutual_information_row": t_m_i_r,
        }

    def _validate_and_normalize_transition_matrix(self):
        T = self.transition_matrix
        if T.ndim != 2 or T.shape[0] != T.shape[1]:
            raise ValueError(f"transition_matrix must be square (got shape {T.shape})")

        if np.any(~np.isfinite(T)):
            raise ValueError("transition_matrix contains NaN/inf")

        if np.any(T < 0):
            raise ValueError("transition_matrix contains negative entries")

        row_sums = T.sum(axis=1, keepdims=True)
     
        zero_rows = (row_sums.squeeze() == 0)
        if np.any(zero_rows):
            T[zero_rows, :] = 1.0 / T.shape[1]
            row_sums = T.sum(axis=1, keepdims=True)

        self.transition_matrix = T / row_sums

    def compute_stationary_distribution(self, tol=1e-12, max_iter=100_000):
        
        T = self.transition_matrix
        n = T.shape[0]

        pi = np.ones(n, dtype=float) / n
        for _ in range(max_iter):
            pi_next = pi @ T
            s = pi_next.sum()
            if s <= 0 or not np.isfinite(s):
                raise ValueError("Failed to compute stationary distribution (degenerate transition matrix).")
            pi_next /= s

            if np.max(np.abs(pi_next - pi)) < tol:
                return pi_next

            pi = pi_next
 
        return pi

    def t_mutual_information(self):
       
        pi = self.stationary_distribution
        T = self.transition_matrix
 
        mask = (T > 0) & (pi[None, :] > 0)   
        Tij = T[mask]
        
        i_idx, j_idx = np.where(mask)
        val = pi[i_idx] * Tij * np.log(Tij / pi[j_idx])
        return float(np.sum(val))

    def t_mutual_information_row(self):
      
        pi = self.stationary_distribution
        T = self.transition_matrix
        n = self.state_number

        out = {}
        for i in range(n):
            mask = (T[i, :] > 0) & (pi > 0)
            Tij = T[i, mask]
            pj = pi[mask]
            out[chr(i + 65)] = float(np.sum(pi[i] * Tij * np.log(Tij / pj)))
        return out

    def t_joint_entropy(self):
      
        pi = self.stationary_distribution
        T = self.transition_matrix

        P = pi[:, None] * T
        mask = P > 0
        return float(-np.sum(P[mask] * np.log(P[mask])))

    def t_conditional_entropy(self):
       
        pi = self.stationary_distribution
        T = self.transition_matrix
        n = self.state_number

        total = 0.0
        for i in range(n):
            mask = T[i, :] > 0
            total += pi[i] * float(-np.sum(T[i, mask] * np.log(T[i, mask])))
        return float(total)

    def t_conditional_entropy_row(self):
       
        T = self.transition_matrix
        n = self.state_number

        out = {}
        for i in range(n):
            mask = T[i, :] > 0
            out[chr(i + 65)] = float(-np.sum(T[i, mask] * np.log(T[i, mask])))
        return out

    def t_stationary_entropy(self):
       
        pi = self.stationary_distribution
        mask = pi > 0
        return float(-np.sum(pi[mask] * np.log(pi[mask])))

# -*- coding: utf-8 -*-

import numpy as np


class FisherKernel:
    
    def __init__(self, input, hmm_model):
        
        self.obs = input
        self.hmm_model = hmm_model

        self.fisher_vector = self.compute_fisher_vectors()


    def compute_fisher_vectors(self):
        """
        Compute a (score-based) Fisher vector from an HMM with Gaussian emissions.
    
        Assumptions / conventions (consistent with your codebase):
        - self.obs is shaped (d, T)
        - c_HMM.evaluate_posterior_moments expects observations shaped (T, d)
          and returns:
            gamma: (T, K)
            xi:    (T, K, K)  with xi[T-1,:,:] left as zeros (unused)
        - hmm_model.centers: (K, d)
        - hmm_model.covars:  (K, d, d)  (full covariances)
        - hmm_model.transition_matrix: (K, K)
    
        Notes:
        - Uses vectorized mean/cov derivatives (much faster).
        - Adds standard Fisher vector normalizations (power + L2).
        - Adds eps-protection for division by tiny transition probabilities.
        """
    
        obs = self.obs  # (d, T)
        if obs.ndim != 2:
            raise ValueError(f"Expected obs with shape (d, T), got {obs.shape}")
    
        # Posterior moments from HMM (expects (T, d))
        gamma, xi = self.hmm_model.reevaluate_moments(obs.T)
    
        K = int(self.hmm_model.n_s)
        T_ = int(obs.shape[1])
        d_ = int(obs.shape[0])
    
        mu = np.asarray(self.hmm_model.centers, dtype=float)          # (K, d)
        covs = np.asarray(self.hmm_model.covars, dtype=float)         # (K, d, d)
        A = np.asarray(self.hmm_model.transition_matrix, dtype=float) # (K, K)
    
        if mu.shape != (K, d_):
            raise ValueError(f"centers shape mismatch: expected {(K, d_)}, got {mu.shape}")
        if covs.shape != (K, d_, d_):
            raise ValueError(f"covars shape mismatch: expected {(K, d_, d_)}, got {covs.shape}")
        if A.shape != (K, K):
            raise ValueError(f"transition_matrix shape mismatch: expected {(K, K)}, got {A.shape}")
    
        fv = []
    
        # ----------------------------
        # (1) d/dA : transition matrix
        # ---------------------------- 
        edges_post = np.sum(xi[:-1], axis=0)  # (K, K)
    
        eps = 1e-12
        dT = np.zeros_like(A)
     
        p_ = np.where(A > 0)
        dT[p_] = edges_post[p_] / (A[p_] + eps)
    
        # Normalize by number of observations  
        fv.append((dT / T_).ravel())
    
        # ---------------------------------------------------
        # (2) d/dmu and d/dSigma : Gaussian emission params
        #     Vectorized, no loops over time t
        # ---------------------------------------------------
        X = obs.T  # (T, d)
        gamma = np.asarray(gamma, dtype=float)  # ensure ndarray
    
        for k in range(K):
            mu_k = mu[k]        # (d,)
            Sigma_k = covs[k]   # (d, d)
     
            Sigma_inv = np.linalg.inv(Sigma_k)
    
            w = gamma[:, k]     # (T,)
            w_sum = float(w.sum())
    
            # Centered observations
            Xc = X - mu_k       # (T, d)
    
            # ---- dMu ----
            # s = sum_t w_t * (x_t - mu): (d,)
            s = (w[:, None] * Xc).sum(axis=0)
            dMu = (Sigma_inv @ s) / T_
            fv.append(dMu.ravel())
    
            # ---- dCov ----
            # scatter = sum_t w_t * (x_t - mu)(x_t - mu)^T : (d, d)
            scatter = Xc.T @ (w[:, None] * Xc)
            dCov = 0.5 * (Sigma_inv @ scatter @ Sigma_inv - w_sum * Sigma_inv) / T_
            fv.append(dCov.ravel())
    
        fv = np.concatenate(fv).astype(float)
    
        # ----------------------------
        # (3) Fisher vector normalizations
        # ----------------------------
        # Power normalization (signed sqrt) + L2 normalization
        fv = np.sign(fv) * np.sqrt(np.abs(fv) + 1e-12)
        fv /= (np.linalg.norm(fv) + 1e-12)
    
        return fv


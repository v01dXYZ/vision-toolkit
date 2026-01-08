# -*- coding: utf-8 -*-

import numpy as np


class FisherKernel:
    
    def __init__(self, input, hmm_model):
        
        self.obs = input
        self.hmm_model = hmm_model

        self.fisher_vector = self.compute_fisher_vectors()


    def compute_fisher_vectors(self, 
                               skip_low_occupancy = True,
                               use_upper_triangular_cov = True, 
                               normalize_transitions_by_row = True,
                               occ_threshold = 1e-8,
                               reg_covar = 1e-6,
                               eps = 1e-12,):
        """
        Compute a (score-based) Fisher vector from an HMM with Gaussian emissions.
    
        Parameters
        ----------
        skip_low_occupancy : bool, default=True
            If True, emission-parameter gradients (dMu, dCov) for a state k are set to 0 when
            its total posterior occupancy w_sum = sum_t gamma[t,k] is below `occ_threshold`.
            This prevents low-occupancy states from contributing a nearly constant covariance
            gradient term (≈ -0.5 * Sigma^{-1} / T), which can act as noise in the FV.
    
        use_upper_triangular_cov : bool, default=True
            If True, keep only the upper-triangular (including diagonal) entries of dCov,
            i.e. d(d+1)/2 values, avoiding redundancy since covariances are symmetric.
            If False, flatten the full d x d matrix.
    
        normalize_transitions_by_row : bool, default=True
            If True, row-normalize the transition gradient block so rows are comparable in scale:
            dT[i,:] /= (sum_j edges_post[i,j] + eps). Independent of the final /T normalization.
    
        occ_threshold : float, default=1e-8
            Occupancy threshold used when `skip_low_occupancy` is enabled.
    
        reg_covar : float, default=1e-6
            Diagonal regularization added to each covariance matrix before inversion:
            Sigma <- (Sigma + Sigma^T)/2 + reg_covar * I. Improves numerical stability.
    
        eps : float, default=1e-12
            Small constant to avoid divide-by-zero and stabilize normalizations.
    
        Assumptions / conventions (consistent with your codebase):
        - self.obs is shaped (d, T)
        - hmm_model.reevaluate_moments expects observations shaped (T, d)
          and returns:
            gamma: (T, K)
            xi:    (T, K, K)  (often xi[T-1,:,:] unused)
        - hmm_model.centers: (K, d)
        - hmm_model.covars:  (K, d, d)  (full covariances)
        - hmm_model.transition_matrix: (K, K)
    
        Notes:
        - Uses vectorized mean/cov derivatives (much faster).
        - Adds standard Fisher vector normalizations (power + L2).
        """
    
        obs = np.asarray(self.obs, dtype=float)
        if obs.ndim != 2:
            raise ValueError(f"Expected obs with shape (d, T), got {obs.shape}")
    
        d_, T_ = int(obs.shape[0]), int(obs.shape[1])
        K = int(self.hmm_model.n_s)
     
        gamma, xi = self.hmm_model.reevaluate_moments(obs.T)
        gamma = np.asarray(gamma, dtype=float)
        xi = np.asarray(xi, dtype=float)
     
        mu = np.asarray(self.hmm_model.centers, dtype=float)          # (K, d)
        covs = np.asarray(self.hmm_model.covars, dtype=float)         # (K, d, d)
        A = np.asarray(self.hmm_model.transition_matrix, dtype=float) # (K, K)
     
        if mu.shape != (K, d_):
            raise ValueError(f"centers shape mismatch: expected {(K, d_)}, got {mu.shape}")
        if covs.shape != (K, d_, d_):
            raise ValueError(f"covars shape mismatch: expected {(K, d_, d_)}, got {covs.shape}")
        if A.shape != (K, K):
            raise ValueError(f"transition_matrix shape mismatch: expected {(K, K)}, got {A.shape}")
        if gamma.shape != (T_, K):
            raise ValueError(f"gamma shape mismatch: expected {(T_, K)}, got {gamma.shape}")
        if xi.ndim != 3 or xi.shape[1:] != (K, K):
            raise ValueError(f"xi shape mismatch: expected (T, K, K), got {xi.shape}")
     
        fv = []
    
        # ---- d/dA : transition matrix ----
        edges_post = np.sum(xi[: max(T_ - 1, 0)], axis=0) if T_ > 1 else np.zeros((K, K))
    
        dT = np.zeros_like(A)
        mask = A > 0
        dT[mask] = edges_post[mask] / (A[mask] + eps)
    
        if normalize_transitions_by_row:
            row_scale = edges_post.sum(axis=1, keepdims=True) + eps
            dT = dT / row_scale
    
        fv.append((dT / max(T_, 1)).ravel())
    
        # ---- d/dmu and d/dSigma : Gaussian emission params ----
        X = obs.T  # (T, d)
        I = np.eye(d_, dtype=float)
    
        triu_idx = np.triu_indices(d_) if use_upper_triangular_cov else None
        cov_feat_len = len(triu_idx[0]) if use_upper_triangular_cov else d_ * d_
    
        for k in range(K):
            mu_k = mu[k]
            Sigma_k = covs[k]
            w = gamma[:, k]          # (T,)
            w_sum = float(w.sum())   # occupancy of state k over the sequence
    
            Xc = X - mu_k
     
            if skip_low_occupancy and w_sum < occ_threshold:
                fv.append(np.zeros(d_, dtype=float))            # dMu
                fv.append(np.zeros(cov_feat_len, dtype=float))  # dCov
                continue
          
            # Regularize & symmetrize covariance
            Sigma_k = 0.5 * (Sigma_k + Sigma_k.T) + reg_covar * I
            Sigma_inv = np.linalg.solve(Sigma_k, I)
    
            # dMu
            s = (w[:, None] * Xc).sum(axis=0)
            dMu = (Sigma_inv @ s) / max(T_, 1)
            fv.append(dMu.ravel())
    
            # dCov
            scatter = Xc.T @ (w[:, None] * Xc)
            dCov = 0.5 * (Sigma_inv @ scatter @ Sigma_inv - w_sum * Sigma_inv) / max(T_, 1)
            dCov = 0.5 * (dCov + dCov.T)
    
            if use_upper_triangular_cov:
                fv.append(dCov[triu_idx])
            else:
                fv.append(dCov.ravel())
    
        fv = np.concatenate(fv).astype(float)
    
        # Fisher vector normalizations
        fv = np.sign(fv) * np.sqrt(np.abs(fv) + eps)
        fv /= (np.linalg.norm(fv) + eps)
    
        return fv
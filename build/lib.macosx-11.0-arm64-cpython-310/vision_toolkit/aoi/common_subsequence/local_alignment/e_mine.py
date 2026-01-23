# -*- coding: utf-8 -*-


import copy

import numpy as np

from vision_toolkit.aoi.aoi_base import AoI_sequences, AoISequence
from vision_toolkit.aoi.common_subsequence.local_alignment.longest_common_subsequence import LongestCommonSubsequence
from vision_toolkit.aoi.global_alignment.levenshtein_distance import LevenshteinDistance
from vision_toolkit.aoi.global_alignment.string_edit_distance import AoI_levenshtein_distance


class eMine:
    def __init__(self, input, config):
        """
        

        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.aoi_sequences = input
        self.config = config
        self.n_sp = len(input)

        d_m = AoI_levenshtein_distance(
            self.aoi_sequences, display_results=False, verbose=False
        )["AoI_levenshtein_distance_matrix"]

        # prevent self-minimum on diagonal
        d_m = d_m.copy()
        d_m += np.diag(np.ones(self.n_sp) * (np.max(d_m) + 1))
        self.d_m = d_m

        self.common_subsequence = self.process_emine()

    def process_emine(self):
        d_m = self.d_m.copy()
        m_ = float(np.max(d_m))

        # Work on a copy to avoid side effects
        aoi_sequences = list(self.aoi_sequences)

        # Sanity: same AoI space
        centers = aoi_sequences[0].centers
        nb_aoi = aoi_sequences[0].nb_aoi
        for s in aoi_sequences[1:]:
            assert s.centers == centers, "All AoISequence must share the same AoI centers for eMine."

        config = copy.deepcopy(self.config)
        config.update({"display_results": False})

        while len(aoi_sequences) > 1:
            i, j = np.unravel_index(np.argmin(d_m), d_m.shape)
            if i > j:
                i, j = j, i

            s_1 = aoi_sequences[i]
            s_2 = aoi_sequences[j]

            lcs = LongestCommonSubsequence([s_1, s_2], config).common_subsequence

            dict_ = {
                "sequence": lcs,
                "durations": None,
                "centers": centers,
                "nb_aoi": nb_aoi,     
                "config": config,
            }
            n_aoi_seq = AoISequence(dict_)

            # Remove by index (largest first)
            del aoi_sequences[j]
            del aoi_sequences[i]

            # Remove rows/cols i,j
            d_m = np.delete(d_m, (i, j), axis=0)
            d_m = np.delete(d_m, (i, j), axis=1)

            # Add new sequence
            aoi_sequences.append(n_aoi_seq)

            # Expand distance matrix with new row/col
            k = len(aoi_sequences) - 1  # index of new sequence
            n_d_m = np.zeros((k + 1, k + 1), dtype=float)

            n_d_m[:-1, :-1] = d_m
            for t in range(k):
                n_d_m[t, k] = LevenshteinDistance([aoi_sequences[t], n_aoi_seq], config).dist_
            n_d_m[k, :k] = n_d_m[:k, k]
            n_d_m[k, k] = m_

            d_m = n_d_m

        return aoi_sequences[0].sequence
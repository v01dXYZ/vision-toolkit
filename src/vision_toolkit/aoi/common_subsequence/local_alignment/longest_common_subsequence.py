# -*- coding: utf-8 -*-

from vision_toolkit.aoi.common_subsequence.local_alignment.c_alignment_algorithms import c_alignment_algorithms as c_alignment


class LongestCommonSubsequence:
    def __init__(self, input, config, id_1="0", id_2="1"):
        """
        

        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.
        id_1 : TYPE, optional
            DESCRIPTION. The default is "0".
        id_2 : TYPE, optional
            DESCRIPTION. The default is "1".

        Returns
        -------
        None.

        """
        self.s_1, self.s_2 = input[0].sequence, input[1].sequence
        self.n_1, self.n_2 = len(self.s_1), len(self.s_2)

        common_subsequence, self.opt_align, lcs_length = c_alignment.longest_common_subsequence(
            self.s_1, self.s_2
        )

        self.common_subsequence = [
            (x[0] if isinstance(x, (list, tuple)) else x) for x in common_subsequence
        ]

        norm_ = config.get("AoI_longest_common_subsequence_normalization", "max")
        denom = max(self.n_1, self.n_2) if norm_ == "max" else min(self.n_1, self.n_2)

        # safe normalization
        if denom == 0:
            self.length_ = 1.0 if (self.n_1 == 0 and self.n_2 == 0) else 0.0
        else:
            self.length_ = float(lcs_length) / float(denom)
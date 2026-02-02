# -*- coding: utf-8 -*-

from collections import Counter

import numpy as np

from vision_toolkit.aoi.aoi_base import AoISequence
from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation
import warnings

class NGram:
    def __init__(self, input, n_w):
        """
        

        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        n_w : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sequence = np.array(input, dtype=object)
        self.n_w = int(n_w)
        self.l_ = len(self.sequence)

        self.table = self.get_frequency()

    def get_frequency(self):
        """
        

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        freq : TYPE
            DESCRIPTION.

        """
        n_w = self.n_w
        l_ = self.l_

        freq = Counter()

        # Guard: too-short sequences or invalid window
        if n_w <= 0:
            raise ValueError("n_w must be a positive integer")
        if l_ < n_w:
            # No n-grams can be formed
            return freq

        denom = (l_ - n_w + 1)
        if denom <= 0:
            return freq

        window_indexer = (
            np.expand_dims(np.arange(n_w), 0)
            + np.expand_dims(np.arange(denom), 0).T
        )

        for subsequence in self.sequence[window_indexer]:
            name = "".join(f"{s}," for s in subsequence)
            freq[name] += 1.0 / denom

        return freq


def AoI_NGram(input, **kwargs):
    
    verbose = kwargs.get("verbose", True)

    if verbose:
        print("Processing NGram Analysis...\n")

    if isinstance(input, AoISequence):
        aoi_sequence = input
        
    elif isinstance(input, (str, BinarySegmentation, Scanpath)):
        aoi_sequence = AoISequence.generate(input, **kwargs)
       
    else:
        raise ValueError(
            "Input must be a str, BinarySegmentation, Scanpath, or AoISequence."
        )

    n_w = int(kwargs.get("AoI_NGram_length", 3))
    n_g = NGram(aoi_sequence.sequence, n_w)

    # Optional: warn if empty because sequence too short
    if verbose and len(n_g.table) == 0 and len(aoi_sequence.sequence) < n_w:
        warnings.warn(
            f"NGram: sequence length ({len(aoi_sequence.sequence)}) < n_w ({n_w}); returning empty histogram.",
            RuntimeWarning,
        )

    results = {"AoI_NGram": n_g.table}

    if verbose:
        print("...NGram Analysis done\n")

    return results


# -*- coding: utf-8 -*-

import copy
import itertools
from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist
import warnings

from vision_toolkit.aoi.aoi_base import AoI_sequences, AoISequence
from vision_toolkit.aoi.pattern_mining.n_gram import NGram
from vision_toolkit.scanpath.similarity.c_comparison_algorithms import c_comparison_algorithms as c_comparison
from vision_toolkit.utils.binning import aoi_dict_dist_mat
from vision_toolkit.visualization.scanpath.similarity.distance_based.elastic import plot_DTW_frechet


class CDBA:
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing CDBA common subsequence...\n")

        assert (
            len(input) > 1 and type(input) == list
        ), "Input must be a list of AoISequence, or a list of Scanpath, or a list of BinarySegmentation, or a list of csv"

        if isinstance(input[0], AoISequence):
            aoi_sequences = input

        else:
            aoi_sequences = AoI_sequences(input, **kwargs)

        self.aoi_sequences = aoi_sequences
        self.centers = aoi_sequences[0].centers
        self.aoi_ = list(self.centers.keys())
        self.d_m, self.i_dict = aoi_dict_dist_mat(self.centers, normalize=False)

        self.config = aoi_sequences[0].config
        self.config.update(
            {
                "AoI_CDBA_initialization_length": kwargs.get(
                    "AoI_CDBA_initialization_length", "min"
                ),
                "AoI_CDBA_initial_random_state": kwargs.get(
                    "AoI_CDBA_initial_random_state", 1
                ),
                "AoI_CDBA_maximum_iterations": kwargs.get(
                    "AoI_CDBA_maximum_iterations", 20
                ),
                "verbose": verbose,
            }
        )

        self.relaxed = False
        self.common_subsequence = self.process_CDBA()

        if verbose:
            print("...CDBA common subsequence done\n")
        self.verbose()

    
    def process_CDBA(self):
        """
        

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        consensus : TYPE
            DESCRIPTION.

        """
        aoi_sequences = self.aoi_sequences
        S_s = [seq.sequence for seq in aoi_sequences]

        # CDBA uses AoI centers to build distances and DTW alignment;
        # therefore all sequences must share the same centers dictionary.
        base_centers = aoi_sequences[0].centers
        for k, seq in enumerate(aoi_sequences[1:], start=1):
            if seq.centers != base_centers:
                raise ValueError(
                    "CDBA requires all AoISequence instances to share the same AoI centers.\n"
                    f"Mismatch detected at sequence index {k}. "
                    "Ensure AoIs are defined in a common reference frame (shared centers)."
                )

        # Bigram candidate set (as a set)  
        # Keys produced by NGram(..., 2) are like "A,B,".
        bi_occ = {key for s_ in S_s for key in NGram(s_, 2).table.keys()}

        # Max occurrences per AoI (constraint 1)  
        counts_ = self.get_counts(S_s)

        # Initialization length 
        init_len = self.config["AoI_CDBA_initialization_length"]
        if init_len == "min":
            l_m = int(np.min([len(s_) for s_ in S_s]))
        elif init_len == "max":
            l_m = int(np.max([len(s_) for s_ in S_s]))
        else:
            raise ValueError("'AoI_CDBA_initialization_length' must be set to 'min' or 'max'")

        # Initialize consensus randomly (reproducible)  
        np.random.seed(self.config.get("AoI_CDBA_initial_random_state", 3))
        consensus = list(np.random.choice(list(self.centers.keys()), l_m))
        old_consensus = copy.deepcopy(consensus)

        # Iterate  
        max_iter = int(self.config.get("AoI_CDBA_maximum_iterations", 20))
        relaxed_count = 0
        iter_ = 0

        while iter_ < max_iter:
            alignments = self.perform_alignments(consensus, S_s)

            # update_consensus uses self.relaxed to indicate whether constraint #2 was relaxed
            consensus = self.update_consensus(alignments, bi_occ, counts_)
            if self.relaxed:
                relaxed_count += 1

            iter_ += 1

            # convergence test
            if consensus == old_consensus:
                break
            old_consensus = copy.deepcopy(consensus)

        # Don't hard-fail when relaxation occurs  
        # In the paper, relaxation can happen when no candidate bigram is available.
        # We surface it as a warning + stored counter.
        self.relaxed_count = relaxed_count
        if relaxed_count > 0:
            warnings.warn(
                f"CDBA: constraint #2 (bigram/candidate-set) had to be relaxed "
                f"{relaxed_count} time(s) during optimization. "
                "This can happen when the bigram constraint is too restrictive or "
                "when initialization length/state leads to dead-ends. "
                "Consider changing AoI_CDBA_initial_random_state or using 'min' length.",
                RuntimeWarning,
            )

        return consensus

    def update_consensus(self, alignments, bi_occ, counts_):
        """
        Same as yours, but bi_occ can now be a set (recommended).
        This version works with either list or set for bi_occ.
        """
        aoi_ = self.aoi_
        i_dict = self.i_dict
        d_m = self.d_m
        relaxed_any = False

        current_counts_ = dict.fromkeys(counts_.keys(), 0)
        consensus = []

        for i in sorted(list(alignments.keys())):
            al_ = alignments[i]

            if i == 0:
                avail = aoi_
                d_t = [
                    np.sum([d_m[i_dict[aoi], i_dict[al]] for al in al_])
                    for aoi in avail
                ]
                opt_aoi = avail[int(np.argmin(d_t))]

            else:
                # constraint 1: do not exceed max occurrences
                avail = [aoi for aoi in aoi_ if current_counts_[aoi] < counts_[aoi]]

                # constraint 2: candidate set / bigram existence
                last = consensus[-1]
                avail2 = [aoi for aoi in avail if f"{last},{aoi}," in bi_occ]

                # relax constraint 2 if needed
                if len(avail2) == 0:
                    relaxed_any = True
                    avail2 = avail
               
                d_t = [
                    np.sum([d_m[i_dict[aoi], i_dict[al]] for al in al_])
                    for aoi in avail2
                ]
                opt_aoi = avail2[int(np.argmin(d_t))]

            consensus.append(opt_aoi)
            current_counts_[opt_aoi] += 1

        self.relaxed = relaxed_any
        
        return consensus
    

    def perform_alignments(self, consensus, S_s):
        """


        Parameters
        ----------
        consensus : TYPE
            DESCRIPTION.
        S_s : TYPE
            DESCRIPTION.

        Returns
        -------
        alignments : TYPE
            DESCRIPTION.

        """

        centers = self.centers
        ## Convert into an array of AoI center positions
        consensus_a = np.array([centers[consensus[i]] for i in range(len(consensus))])
        ## Initialize a dictionary to keep AoI aligned with each element from
        ## the consensus sequence
        alignments = dict.fromkeys(range(len(consensus)), [])

        for k in range(len(S_s)):
            s_ = np.array(S_s[k])
            s_a = np.array([centers[s_[i]] for i in range(len(s_))])
            d_m = cdist(consensus_a, s_a, metric="euclidean")
            opt_links, dist_ = c_comparison.DTW(consensus_a.T, s_a.T, d_m)
            for i in range(len(consensus)):
                ## Find indexes of alignments which involve consensus[i]
                idx = np.argwhere((opt_links[:, 2, 0]) == i)[:, 0]
                ## Find elements from s_ aligned with consensus[i]
                al_idx = opt_links[idx, 2, 1].astype(int)
                al_ = s_[al_idx]

                ## Update AoI aligned with each element from the consensus sequence
                ## Note that several element AoI from one sequence can be aligned
                ## with each element from the consensus sequence
                alignments[i] = alignments[i] + list(al_)

        return alignments

    def get_counts(self, S_s):
        """


        Parameters
        ----------
        S_s : TYPE
            DESCRIPTION.

        Returns
        -------
        counts_ : TYPE
            DESCRIPTION.

        """

        aoi_ = self.aoi_
        counts_ = dict()

        for aoi in aoi_:
            c_ = [s_.count(aoi) for s_ in S_s]
            counts_.update({aoi: max(c_)})

        return counts_

    def verbose(self, add_=None):
        """


        Parameters
        ----------
        add_ : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        if self.config["verbose"]:
            print("\n --- Config used: ---\n")

            for it in self.config.keys():
                print(
                    "# {it}:{esp}{val}".format(
                        it=it, esp=" " * (50 - len(it)), val=self.config[it]
                    )
                )
            if add_ is not None:
                for it in add_.keys():
                    print(
                        "# {it}:{esp}{val}".format(
                            it=it, esp=" " * (50 - len(it)), val=add_[it]
                        )
                    )
            print("\n")


def AoI_CDBA(input, **kwargs):
    
    cdba = CDBA(input, **kwargs)
    results = dict({"AoI_CDBA_common_subsequence": cdba.common_subsequence})

    return results







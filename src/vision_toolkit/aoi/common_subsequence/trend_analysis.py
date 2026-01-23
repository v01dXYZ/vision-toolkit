# -*- coding: utf-8 -*-

import copy
import operator
import re
from itertools import groupby

import numpy as np

from vision_toolkit.aoi.aoi_base import AoI_sequences, AoISequence

np.random.seed(15)


class TrendAnalysis:
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
            print("Processing AoI String Edit Distance...\n")

        assert (
            len(input) > 1 and type(input) == list
        ), "Input must be a list of AoISequence, or a list of Scanpath, or a list of BinarySegmentation, or a list of csv"

        if isinstance(input[0], AoISequence):
            aoi_sequences = input

        else:
            aoi_sequences = AoI_sequences(input, **kwargs)

        self.config = aoi_sequences[0].config
        self.config.update(
            {
                "AoI_trend_analysis_tolerance_level": kwargs.get(
                    "AoI_trend_analysis_tolerance_level", 0.95
                ),
                "verbose": verbose,
            }
        )
        self.aoi_sequences = aoi_sequences

        self.common_subsequence = self.process_trend_analysis()
        self.verbose()

    def process_trend_analysis(self):
        """


        Returns
        -------
        trending_sequence : TYPE
            DESCRIPTION.

        """

        ## Initialize lists of AoI simplified sequences and concatenated AoI
        ## simplified sequence
        S_s, S_s_c = [], []
        ## Initialize lists of AoI simplified sequence durations and concatenated
        ## AoI simplified sequence durations
        D_s, D_s_c = [], []

        for i in range(len(self.aoi_sequences)):
            assert (
                self.aoi_sequences[i].durations is not None
            ), "AoI_durations must be provided to perform TrendAnalysis"

            s_s, d_s = self.simplify(
                self.aoi_sequences[i].sequence, self.aoi_sequences[i].durations
            )
            S_s.append(s_s)
            S_s_c += s_s
            D_s.append(d_s)
            D_s_c += d_s

        ## Get individual instances from the concatenated AoI sequence
        i_inst = sorted(list(set(S_s_c)))

        ## Compute list of shared instances, wrt the tolerance_level parameter
        shr_inst = self.get_shared(i_inst, S_s)

        ## Compute attentionnal thresholds from fully shared instances
        n_t, d_t = self.get_importance_thresholds(
            np.array(S_s_c), np.array(D_s_c), shr_inst
        )

        ## Compute list of other instance candidates
        c_inst = list(set(i_inst) - set(shr_inst))

        ## Remove unfrequent and short candidate instances
        c_inst = self.remove_candidates(
            c_inst, np.array(S_s_c), np.array(D_s_c), n_t, d_t
        )
        n_S_s, n_D_s = self.remove_instances(S_s, D_s, shr_inst, c_inst)
        trending_sequence = self.comp_trending_sequence(n_S_s, n_D_s, shr_inst, c_inst)

        return trending_sequence


    def comp_trending_sequence(self, n_S_s, n_D_s, shr_inst, c_inst):
        """
        

        Parameters
        ----------
        n_S_s : TYPE
            DESCRIPTION.
        n_D_s : TYPE
            DESCRIPTION.
        shr_inst : TYPE
            DESCRIPTION.
        c_inst : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # All instances considered
        i_inst = sorted(set(shr_inst + c_inst))
    
        # helper: STA priority for an instance at position P in a sequence of length L  
        # Paper: ψ = 1 − P * z, where z = (1 - 0.1)/(L - 1)  (maxi=1, mini=0.1) :contentReference[oaicite:3]{index=3}
        def psi(P, L, maxi=1.0, mini=0.1):
            if L <= 1:
                return maxi
            z = (maxi - mini) / (L - 1)
            return maxi - P * z
    
        # Compute total priority, total duration, total occurrences per instance 
        total_priority = {inst: 0.0 for inst in i_inst}
        total_dur = {inst: 0.0 for inst in i_inst}
        total_occ = {inst: 0.0 for inst in i_inst}
    
        for seq, dur in zip(n_S_s, n_D_s):
            # seq: list of instance labels; dur: list [[dur, count], ...] aligned with seq
            L = len(seq)
            for pos, inst in enumerate(seq):
                if inst in total_priority:
                    total_priority[inst] += psi(pos, L)
                    total_dur[inst] += float(dur[pos][0])
                    total_occ[inst] += float(dur[pos][1])
    
        # Threshold: minimum priority among shared instances 
        if len(shr_inst) > 0:
            p_t = min(total_priority[inst] for inst in shr_inst)
        else:
            # edge-case: no shared inst -> keep everything by priority only
            p_t = float("-inf")
    
        # Sort by (priority desc, duration desc, occurrences desc) 
        def sort_key(inst):
            return (total_priority[inst], total_dur[inst], total_occ[inst])
    
        sorted_inst = sorted(i_inst, key=sort_key, reverse=True)
    
        # Build trending sequence with rule "shared always; candidates only if >= p_t" 
        t_sp = []
        for inst in sorted_inst:
            if inst in shr_inst:
                t_sp.append(re.split(r"(\d+)", inst)[0])  # drop numbering
            else:
                if total_priority[inst] >= p_t:
                    t_sp.append(re.split(r"(\d+)", inst)[0])
    
        # Remove consecutive duplicates (STA final abstraction step)  
        t_sp = [k for k, _ in groupby(t_sp)]
        return t_sp


    def remove_instances(self, S_s, D_s, shr_inst, c_inst):
        """
        
        

        Parameters
        ----------
        S_s : TYPE
            DESCRIPTION.
        D_s : TYPE
            DESCRIPTION.
        shr_inst : TYPE
            DESCRIPTION.
        c_inst : TYPE
            DESCRIPTION.

        Returns
        -------
        n_S_s : TYPE
            DESCRIPTION.
        n_D_s : TYPE
            DESCRIPTION.

        """
        n_S_s, n_D_s = [], []
    
        for s_s, d_s in zip(S_s, D_s):
            keep_s, keep_d = [], []
            for inst, inst_d in zip(s_s, d_s):
                if inst in shr_inst or inst in c_inst:
                    keep_s.append(inst)
                    keep_d.append(inst_d)
            n_S_s.append(keep_s)
            n_D_s.append(keep_d)
    
        return n_S_s, n_D_s


    def remove_candidates(self, c_inst, S_s_c, D_s_c, n_t, d_t):
        """


        Parameters
        ----------
        c_inst : TYPE
            DESCRIPTION.
        S_s_c : TYPE
            DESCRIPTION.
        D_s_c : TYPE
            DESCRIPTION.
        n_t : TYPE
            DESCRIPTION.
        d_t : TYPE
            DESCRIPTION.

        Returns
        -------
        n_c_inst : TYPE
            DESCRIPTION.

        """

        ## Get candidates instances and initiate new candidate instance list
        c_inst = c_inst
        n_c_inst = []

        for inst in c_inst:
            idx = np.where(S_s_c == inst)[0]
            ## Get total number of occurences on the candidate instance
            if np.sum(D_s_c[idx, 1]) >= n_t:
                ## Get total duration of occurences on the candidate instance
                if np.sum(D_s_c[idx, 0]) >= d_t:
                    n_c_inst.append(inst)

        return n_c_inst

    def get_importance_thresholds(self, S_s_c, D_s_c, shr_inst):
        """


        Parameters
        ----------
        S_s_c : TYPE
            DESCRIPTION.
        D_s_c : TYPE
            DESCRIPTION.
        shr_inst : TYPE
            DESCRIPTION.

        Returns
        -------
        min_n : TYPE
            DESCRIPTION.
        min_d : TYPE
            DESCRIPTION.

        """
        if len(shr_inst) == 0:
            return 0, 0
        
        ## Find instance indexes corresponding to shared element instances
        idx_c = [np.where(S_s_c == inst)[0] for inst in shr_inst]

        ## Compute the minimum total duration of the shared instances
        min_d = min([np.sum(D_s_c[idx, 0]) for idx in idx_c])

        ## Compute the minimum total number of occurrences for the shared instances
        min_n = min([np.sum(D_s_c[idx, 1]) for idx in idx_c])

        return min_n, min_d

    def get_shared(self, i_inst, S_s):
        """


        Parameters
        ----------
        i_inst : TYPE
            DESCRIPTION.
        S_s : TYPE
            DESCRIPTION.

        Returns
        -------
        shr_inst : TYPE
            DESCRIPTION.

        """
        t_l = self.config["AoI_trend_analysis_tolerance_level"]
        shr_inst = []

        for inst in i_inst:
            ## Count the number of sequences with this instance
            in_ = 0
            for s_ in S_s:
                in_ += inst in s_

            if in_ / len(S_s) >= t_l:
                shr_inst.append(inst)

        return shr_inst


    def simplify(self, s_, d_):
        """
        Collapse consecutive repetitions and build "instances" of each AoI.
        Each instance is assigned an ID like 'A1', 'A2', ... where numbering
        follows decreasing instance duration for the same AoI.
        Returns:
            - list of instance IDs (object strings)
            - list of [instance_total_duration, instance_fixation_count]
        """
        if s_ is None or len(s_) == 0:
            return [], []
        
        # 1) Collapse consecutive repetitions
        s_s = [s_[0]]
        d_s = [[d_[0], 1]]  # [total_duration, fixation_count]
    
        for i in range(1, len(s_)):
            if s_[i] == s_[i - 1]:
                d_s[-1][0] += d_[i]
                d_s[-1][1] += 1
            else:
                s_s.append(s_[i])
                d_s.append([d_[i], 1])
    
        s_s = np.array(s_s, dtype=object)
        d_s = np.array(d_s, dtype=float)
    
        # 2) Create instance labels per AoI by decreasing duration
        n_s_s = np.empty_like(s_s, dtype=object)
        for aoi in sorted(set(s_s)):
            idx = np.where(s_s == aoi)[0]              # positions of this AoI in simplified seq
            durations = d_s[idx, 0]                    # durations of those instances
    
            # indices of idx sorted by decreasing duration
            order = np.argsort(-durations)             # local positions 0..len(idx)-1
    
            # Assign ranks starting at 1 (A1 = longest, A2 = second, ...)
            for rank, local_pos in enumerate(order, start=1):
                inst_index = idx[local_pos]            # actual index in s_s / d_s
                n_s_s[inst_index] = f"{s_s[inst_index]}{rank}"
    
        return list(n_s_s), list(d_s)
    

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


def AoI_trend_analysis(input, **kwargs):
    
    ta = TrendAnalysis(input, **kwargs)
    results = dict({"AoI_trend_analysis_common_subsequence": ta.common_subsequence})

    return results




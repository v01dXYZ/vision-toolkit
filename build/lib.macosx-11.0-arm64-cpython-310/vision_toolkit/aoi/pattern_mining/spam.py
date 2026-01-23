# -*- coding: utf-8 -*-

import copy
import sys
from bisect import bisect
from math import ceil 
from typing import List, Dict, Any, Optional, Tuple

from bitarray import bitarray
from bisect import bisect_right

from vision_toolkit.aoi.aoi_base import AoI_sequences, AoISequence 

  
def collapse_consecutive(seq: List[Any]) -> List[Any]:
    """Collapse consecutive repetitions: A A A B B A -> A B A"""
    if not seq:
        return []
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out


class BitmapSeq:
    """
    Bitmap for *sequential* SPAM (S-step only).
    Bits represent itemset/time positions in the concatenated database.
    """

    def __init__(self, last_bit_index: int):
        # use python list of bool for simplicity (replace with bitarray if you want speed)
        self.bitmap = [False] * (last_bit_index + 1)
        self.support = 0
        self.last_sid = -1  # last sequence id counted for support

    def set_bit(self, idx: int):
        self.bitmap[idx] = True

    def search(self, value: bool = True) -> List[int]:
        return [i for i, v in enumerate(self.bitmap) if v is value]

    @staticmethod
    def bit_to_sid(bit: int, sequences_offsets: List[int]) -> int:
        # sequences_offsets is cumulative start indices per sid in the concatenated timeline
        # find rightmost offset <= bit
        # linear would be ok for small, but keep binary:
        sid = bisect_right(sequences_offsets, bit) - 1
        return max(sid, 0)

    @staticmethod
    def last_bit_of_sid(sid: int, sequences_offsets: List[int], last_bit_index: int) -> int:
        if sid + 1 >= len(sequences_offsets):
            return last_bit_index
        return sequences_offsets[sid + 1] - 1

    def register_occurrence(self, sid: int, tid: int, sequences_offsets: List[int]):
        """
        Set bit for (sid, tid) occurrence and update support once per sid.
        """
        pos = sequences_offsets[sid] + tid
        self.bitmap[pos] = True
        if sid != self.last_sid:
            self.support += 1
            self.last_sid = sid

    def s_step(self, next_item_bitmap: "BitmapSeq", sequences_offsets: List[int], last_bit_index: int) -> "BitmapSeq":
        """
        Correct S-step (subsequence with gaps):
        For each sequence, consider ALL possible end positions of prefix.
        Allow next symbol matches strictly AFTER each end position in that same sequence.
        """
        new_bm = BitmapSeq(last_bit_index)

        prefix_bits = [i for i, v in enumerate(self.bitmap) if v]
        next_bits = [i for i, v in enumerate(next_item_bitmap.bitmap) if v]

        if not prefix_bits or not next_bits:
            return new_bm

        # group prefix bits by sid
        start = 0
        n_next = len(next_bits)
        idx_next_global = 0
        last_sid_counted = -1

        while start < len(prefix_bits):
            first_bit = prefix_bits[start]
            sid = self.bit_to_sid(first_bit, sequences_offsets)
            lb = self.last_bit_of_sid(sid, sequences_offsets, last_bit_index)

            end = start
            while end < len(prefix_bits):
                if self.bit_to_sid(prefix_bits[end], sequences_offsets) != sid:
                    break
                end += 1

            ends_this_sid = prefix_bits[start:end]

            # optional: advance global pointer to be near this sid segment
            sid_start = sequences_offsets[sid]
            while idx_next_global < n_next and next_bits[idx_next_global] < sid_start:
                idx_next_global += 1

            matched = False
            for end_bit in ends_this_sid:
                j = bisect_right(next_bits, end_bit, lo=idx_next_global)
                while j < n_next:
                    b = next_bits[j]
                    if b > lb:
                        break
                    new_bm.set_bit(b)
                    matched = True
                    j += 1

            if matched and sid != last_sid_counted:
                new_bm.support += 1
                last_sid_counted = sid
                new_bm.last_sid = sid

            start = end

        return new_bm


class SpamSeqAlgo:
    """
    SPAM (S-step only) with:
      - optional collapse consecutive duplicates per sequence
      - top-k by support
    Patterns are returned as simple lists of symbols: ['D','B','E'].
    """

    def __init__(self, min_sup_rel: float = 0.5, max_pattern_length: int = 50):
        self.min_sup_rel = float(min_sup_rel)
        self.max_pattern_length = int(max_pattern_length)

        self.vertical_db: Dict[Any, BitmapSeq] = {}
        self.sequences_offsets: List[int] = []
        self.last_bit_index: int = -1
        self.min_sup: int = 1

        # Collected patterns: {tuple(pattern): support}
        self.pattern_support: Dict[Tuple[Any, ...], int] = {}

    def fit(self, sequences: List[List[Any]]):
        self._build_db(sequences)
        self._mine()

    def _build_db(self, sequences: List[List[Any]]):
        # offsets and last_bit_index
        bit_index = 0
        self.sequences_offsets = [0]
        for sid, seq in enumerate(sequences):
            bit_index += len(seq)
            if sid < len(sequences) - 1:
                self.sequences_offsets.append(bit_index)
        self.last_bit_index = bit_index - 1

        # min support (absolute)
        self.min_sup = max(ceil(self.min_sup_rel * len(sequences)), 1)

        # vertical db: per symbol bitmap of occurrences
        self.vertical_db = {}
        for sid, seq in enumerate(sequences):
            for tid, sym in enumerate(seq):
                bm = self.vertical_db.get(sym)
                if bm is None:
                    bm = BitmapSeq(self.last_bit_index)
                    self.vertical_db[sym] = bm
                bm.register_occurrence(sid, tid, self.sequences_offsets)

        # remove infrequent 1-items
        for sym in list(self.vertical_db.keys()):
            if self.vertical_db[sym].support < self.min_sup:
                del self.vertical_db[sym]

    def _mine(self):
        items = sorted(self.vertical_db.keys(), key=lambda x: str(x))
        for sym in items:
            bm = self.vertical_db[sym]
            pat = (sym,)
            self.pattern_support[pat] = bm.support
            if self.max_pattern_length > 1:
                self._dfs(prefix=pat, prefix_bm=bm, items=items, depth=1)

    def _dfs(self, prefix: Tuple[Any, ...], prefix_bm: BitmapSeq, items: List[Any], depth: int):
        if depth >= self.max_pattern_length:
            return

        for sym in items:
            next_bm = prefix_bm.s_step(self.vertical_db[sym], self.sequences_offsets, self.last_bit_index)
            if next_bm.support >= self.min_sup:
                new_prefix = prefix + (sym,)
                self.pattern_support[new_prefix] = next_bm.support
                self._dfs(new_prefix, next_bm, items, depth + 1)

    def get_top_k(self, k: int = 50) -> List[Dict[str, Any]]:
        # sort by support desc, then length desc, then lexicographic
        sorted_pats = sorted(
            self.pattern_support.items(),
            key=lambda kv: (kv[1], len(kv[0]), kv[0]),
            reverse=True,
        )
        out = []
        for pat, sup in sorted_pats[:k]:
            out.append({"pattern": list(pat), "support": int(sup)})
        return out


class AoISPAM:
    def __init__(self, input, **kwargs):
        """
        Parameters (defaults):
          - AoI_SPAM_support: 0.5
          - AoI_SPAM_top_k: 50
          - AoI_SPAM_collapse: True
          - AoI_SPAM_max_length: 50  (also defaults to top_k if you want; here fixed 50)
        """
        verbose = kwargs.get("verbose", True)
        if verbose:
            print("Processing AoI SPAM (sequential, S-step only)...\n")

        assert len(input) > 1 and type(input) == list, (
            "Input must be a list of AoISequence, or a list of Scanpath, or a list of "
            "BinarySegmentation, or a list of csv"
        )

        # Build AoI sequences list
        if isinstance(input[0], AoISequence):
            aoi_sequences = input
        else:
            aoi_sequences = AoI_sequences(input, **kwargs)

        self.config = aoi_sequences[0].config
        self.config.update(
            {
                "AoI_SPAM_support": kwargs.get("AoI_SPAM_support", 0.5),
                "AoI_SPAM_top_k": kwargs.get("AoI_SPAM_top_k", 50),
                "AoI_SPAM_collapse": kwargs.get("AoI_SPAM_collapse", True),
                "AoI_SPAM_max_length": kwargs.get("AoI_SPAM_max_length", 50),
                "verbose": verbose,
            }
        )
        self.aoi_sequences = aoi_sequences

        # Prepare sequences (as lists of symbols)
        sequences = [list(aoi.sequence) for aoi in aoi_sequences]
        if self.config["AoI_SPAM_collapse"]:
            sequences = [collapse_consecutive(seq) for seq in sequences]

        algo = SpamSeqAlgo(
            min_sup_rel=self.config["AoI_SPAM_support"],
            max_pattern_length=self.config["AoI_SPAM_max_length"],
        )
        algo.fit(sequences)
        self.frequent_sequences = algo.get_top_k(self.config["AoI_SPAM_top_k"])

        if verbose:
            print("...AoI SPAM done\n")

 
def AoI_SPAM(input, **kwargs):
    
    sp = AoISPAM(input, **kwargs)
    results = {"AoI_SPAM_top_k_sequences": sp.frequent_sequences}
    
    return results











# -*- coding: utf-8 -*-

from vision_toolkit.aoi.aoi_base import AoISequence 



from lempel_ziv_complexity import (lempel_ziv_complexity,
                                   lempel_ziv_decomposition)


class LemplZiv:
    def __init__(self, input, **kwargs):
        
        self.seq = input.sequence

        lzc_, dec_ = self.process()
        self.results = dict(
            {"AoI_lempel_ziv_complexity": lzc_, "AoI_lempel_ziv_decomposition": dec_}
        )

    def process(self):
        
        seq = self.seq
        seq = "".join(seq)
        lzc_ = lempel_ziv_complexity(seq)
        dec_ = lempel_ziv_decomposition(seq)

        return lzc_, dec_


def AoI_lempel_ziv(input, **kwargs):
    
    if isinstance(input, AoISequence):
        aoi_sequence = input
    else:
        aoi_sequence = AoISequence(input, **kwargs)
    
    lz = LemplZiv(aoi_sequence, **kwargs)
    results = lz.results

    return results

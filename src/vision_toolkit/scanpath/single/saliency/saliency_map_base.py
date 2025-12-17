# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal

from scipy.ndimage import convolve  
import matplotlib.pyplot as plt 

from vision_toolkit.utils.binning import spatial_bin
from vision_toolkit.visualization.scanpath.single.saliency import plot_saliency_map

from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation

 
 
class SaliencyMap:

    def __init__(self,
                 input,
                 comp_saliency_map=True,
                 **kwargs):
   
        verbose = kwargs.get("verbose", True)
        display_results = kwargs.get("display_results", True)
        display_path = kwargs.get("display_path", None)

        std = kwargs.get("scanpath_saliency_gaussian_std", 3)
        k_l = kwargs.get("scanpath_saliency_kernel_length", 50)
        pixel_number_x = kwargs.get("scanpath_saliency_pixel_number_x", 100)
        pixel_number_y = kwargs.get("scanpath_saliency_pixel_number_y", None)
 
        if isinstance(input, list):
            if isinstance(input[0], str):
                scanpaths = [Scanpath.generate(inp, **kwargs) for inp in input]
            elif isinstance(input[0], BinarySegmentation):
                scanpaths = [Scanpath.generate(inp, **kwargs) for inp in input]
            elif isinstance(input[0], Scanpath):
                scanpaths = input
            else:
                raise ValueError(
                    "Input must be a list of Scanpath, BinarySegmentation or csv"
                )
        else:
            if isinstance(input, str):
                scanpaths = [Scanpath.generate(input, **kwargs)]
            elif isinstance(input, BinarySegmentation):
                scanpaths = [Scanpath.generate(input, **kwargs)]
            elif isinstance(input, Scanpath):
                scanpaths = [input]
            else:
                raise ValueError(
                    "Input must be a Scanpath, a BinarySegmentation, or a csv"
                )

        self.scanpaths = scanpaths

        self.size_plan_x = self.scanpaths[0].config["size_plan_x"]
        self.size_plan_y = self.scanpaths[0].config["size_plan_y"]

        ratio = self.size_plan_x / self.size_plan_y
        if pixel_number_y is None:
            pixel_number_y = int(pixel_number_x / ratio)
 
        self.scanpaths[0].config.update(
            {
                "scanpath_saliency_gaussian_std": std,
                "scanpath_saliency_pixel_number_x": pixel_number_x,
                "scanpath_saliency_pixel_number_y": pixel_number_y,
                "verbose": verbose,
                "display_results": display_results,
                "display_path": display_path,
            }
        )

        self.saliency_map = None
 
        for scanpath in self.scanpaths:
            assert (
                scanpath.config["size_plan_x"] == self.size_plan_x
            ), 'All recordings must have the same "size_plan_x"'
            assert (
                scanpath.config["size_plan_y"] == self.size_plan_y
            ), 'All recordings must have the same "size_plan_y"'
 
        self.p_n_x = pixel_number_x + 1 if (pixel_number_x % 2) == 0 else pixel_number_x
        self.p_n_y = pixel_number_y + 1 if (pixel_number_y % 2) == 0 else pixel_number_y

        self.std = std
        self.k_l = k_l
 
        self.s_b = []
        for sp in self.scanpaths:
            seq = sp.values
            self.s_b.append(
                spatial_bin(
                    seq[0:2],
                    self.p_n_x,
                    self.p_n_y,
                    self.size_plan_x,
                    self.size_plan_y,
                )
            )
 
        if comp_saliency_map:
            self.saliency_map = self.comp_saliency_map(self.s_b)

            if display_results:
                plot_saliency_map(self.saliency_map, display_path)

        self.scanpaths[0].verbose()

    def comp_saliency_map(self, s_b):
   
        f_m = np.zeros((self.p_n_y, self.p_n_x), dtype=float)
        for s in s_b:
            l_f_m = np.zeros_like(f_m)
 
            unique, counts = np.unique(s[0:2], return_counts=True, axis=1)

            for i, coord in enumerate(unique.T):
                x = int(coord[0])
                y = int(coord[1])

                if 0 <= x < self.p_n_x and 0 <= y < self.p_n_y:
                    l_f_m[y, x] = counts[i]

            f_m += l_f_m
 
        f_m = f_m / len(s_b)
 
        def gkern(kernlen=self.k_l, std=self.std):
            try:
                from scipy.signal.windows import gaussian as _gauss
            except ImportError:
                try:
                    from scipy.signal import gaussian as _gauss
                except ImportError:
                    _gauss = None

            if _gauss is not None:
                g1d = _gauss(kernlen, std).reshape(kernlen, 1)
            else:
                x = np.arange(kernlen) - (kernlen - 1) / 2.0
                g1d = np.exp(-(x**2) / (2 * std**2)).reshape(kernlen, 1)

            g2d = np.outer(g1d, g1d)
            g2d /= g2d.sum()
            return g2d
 
        s_m = signal.convolve2d(f_m, gkern(), mode="same", boundary="symm")

        total = np.sum(s_m)
        if total > 0:
            s_m = s_m / total
        else:
            s_m = np.zeros_like(s_m)

        return s_m
  
    
 
def scanpath_saliency_map(input, **kwargs):
   
    saliency_map_i = SaliencyMap(input, **kwargs)
    results = dict({"salency_map": saliency_map_i.saliency_map})

    return results
       
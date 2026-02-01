# -*- coding: utf-8 -*-

import vision_toolkit as v
 

root = 'dataset/'

 
sp1 = v.Scanpath(root + 'data_1.csv', 
                sampling_frequency = 256,  
                segmentation_method = 'I_HMM',
                distance_type = 'angular',                        
                display_segmentation = True,
                size_plan_x = 1200,
                size_plan_y = 800,
                display_scanpath=True)
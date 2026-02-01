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

sp2 = v.Scanpath(root + 'data_2.csv', 
                sampling_frequency = 256,  
                segmentation_method = 'I_HMM',
                distance_type = 'angular',                        
                display_segmentation = True,
                size_plan_x = 1200,
                size_plan_y = 800,
                display_scanpath=True)


print(v.scanpath_CRQA_recurrence_rate([sp1,sp2]))
print(v.scanpath_CRQA_laminarity([sp1,sp2]))
print(v.scanpath_CRQA_determinism([sp1,sp2]))
print(v.scanpath_CRQA_entropy([sp1,sp2]))
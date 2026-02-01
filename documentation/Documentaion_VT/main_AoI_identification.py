# -*- coding: utf-8 -*-

import vision_toolkit as v
 

data = 'dataset/nat006.csv' 
image_ref = 'dataset/nat006.bmp'

bs = v.BinarySegmentation(data, 
                          sampling_frequency = 500,  
                          segmentation_method = 'I_HMM',
                          distance_type = 'euclidean',                        
                          display_segmentation = False,
                          verbose=False,
                          size_plan_x = 921,
                          size_plan_y = 630,  
                          )

sc = v.Scanpath(bs, 
                ref_image=image_ref,
                display_scanpath=True,
                display_scanpath_path='figures/school',
                verbose=False)




aoi_ms = v.AoISequence(bs, 
                      ref_image=image_ref,
                      AoI_identification_method='I_MS',  
                      verbose=False,
                      AoI_MS_cluster_all=True)

aoi_dt = v.AoISequence(bs, 
                      ref_image=image_ref,
                      AoI_identification_method='I_DT',  
                      verbose=False,
                      AoI_IDT_reassign_noise=False)

aoi_km = v.AoISequence(bs, 
                      ref_image=image_ref,
                      AoI_identification_method='I_KM',  
                      verbose=False)

aoi_dp = v.AoISequence(bs, 
                      ref_image=image_ref,
                      AoI_identification_method='I_DP',  
                      verbose=False)

aoi_ap = v.AoISequence(bs, 
                      ref_image=image_ref,
                      AoI_identification_method='I_AP',  
                      verbose=False)
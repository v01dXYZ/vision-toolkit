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

sm1 = v.scanpath_saliency_map(sp1)
sm2 = v.scanpath_saliency_map(sp2)


print(v.scanpath_saliency_pearson_corr([sm1,sm2], 
                                       size_plan_x=1200,
                                       size_plan_y=800))
print(v.scanpath_saliency_kl_divergence([sm1,sm2], 
                                       size_plan_x=1200,
                                       size_plan_y=800))

print(v.scanpath_saliency_percentile(sp1, sm1))
print(v.scanpath_saliency_nss(sp1, sm1))
print(v.scanpath_saliency_information_gain(sp1, sm1))
print(v.scanpath_saliency_auc_judd(sp1, sm1))
print(v.scanpath_saliency_auc_borji(sp1, sm1))
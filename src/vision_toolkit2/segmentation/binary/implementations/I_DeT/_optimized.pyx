# distutils: language = c++

import numpy as np

from vision_toolkit2.segmentation.utils import centroids_from_ints, interval_merging

from libc cimport math

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.pair cimport pair as cpair 
 

cdef map[int, int] dict_to_cmap(dict p_dict):
    
    cdef int map_key
    cdef int map_val
     
    cdef cpair[int, int] map_e 
    cdef map[int, int] c_map
    
    for key,val in p_dict.items():
        
        map_key = key
        map_val = val   
        map_e = (map_key, map_val)
        c_map.insert(map_e)
        
    return c_map

     
cdef expand_cluster (double[:,:] g_npts, bool euclidean, 
                     int n_s, int idx, 
                     list neigh, double d_t, 
                     int n_w, dict avlb):
     
    cdef list l_C_clus = [idx]
    cdef list n_neigh = []
    
    for neigh_idx in neigh:

        if avlb[neigh_idx] == True:
            
            l_C_clus.append(neigh_idx)
            avlb[neigh_idx] = False
                
        n_neigh = vareps_neighborhood (g_npts, euclidean,
                                       n_s, neigh_idx, 
                                       d_t, n_w)    
        
        if len(n_neigh) > n_w:
             
            for key in n_neigh :
                
                if key not in neigh:
                    neigh.append(key)
     
    return l_C_clus, avlb


cdef list vareps_neighborhood (double[:,:] g_npts, 
                               bool euclidean, 
                               int n_s, int idx, 
                               double d_t, int n_w):
 
    cdef list neigh = []
    cdef double[:] ref_g_npts = g_npts[:,idx]

    #to the right
    cdef double d_r = 0.0
    cdef int r = idx
    
    cdef double d_l = 0.0
    cdef int l = idx
    
    cdef double n_1 = math.sqrt(ref_g_npts[0]**2 
                                + ref_g_npts[1]**2 
                                + ref_g_npts[2]**2)
    cdef double n_2 = 0.0
    
    cdef double ad_r = 0.0
    cdef double ad_d = 0.0
 
    if euclidean == True:
        
        with nogil:
            
            while r+1 < min(n_s, idx+n_w+1):  
                if d_r < d_t: 
                    
                    r = r+1  
                    d_r = math.sqrt((ref_g_npts[0]-g_npts[0,r])**2 
                                    + (ref_g_npts[1]-g_npts[1,r])**2 
                                    + (ref_g_npts[2]-g_npts[2,r])**2)
                else:
                    break
            
    else:
        
        with nogil:
             
            while r+1 < min(n_s, idx+n_w+1):   
                if d_r < d_t: 
            
                    r = r+1   
                    n_2 = math.sqrt(g_npts[0,r]**2 
                                    + g_npts[1,r]**2 
                                    + g_npts[2,r]**2)
                    
                    ad_r = math.acos((ref_g_npts[0]*g_npts[0,r]
                                      + ref_g_npts[1]*g_npts[1,r]
                                      + ref_g_npts[2]*g_npts[2,r]) 
                                     / (n_1*n_2))
                    d_r = math.fabs(ad_r/(2*math.pi)*360)
                    
                else:
                    break
    
    #to the left 
    if euclidean == True:
        
        with nogil:
            
            while l > max(0, idx-n_w-1):
                
                if d_l < d_t: 
         
                    l -= 1 
                    d_l = math.sqrt((ref_g_npts[0]-g_npts[0,l])**2 
                                    + (ref_g_npts[1]-g_npts[1,l])**2 
                                    + (ref_g_npts[2]-g_npts[2,l])**2)
                else:
                    break
            
    else:
        
        with nogil:
            
            while l > max(0, idx-n_w-1):
                
                if d_l < d_t: 
         
                    l -= 1 
                    n_2 = math.sqrt(g_npts[0,l]**2 
                                    + g_npts[1,l]**2 
                                    + g_npts[2,l]**2)
                    
                    ad_r = math.acos((ref_g_npts[0]*g_npts[0,l]
                                      + ref_g_npts[1]*g_npts[1,l]
                                      + ref_g_npts[2]*g_npts[2,l]) 
                                     / (n_1*n_2))
                    d_l = math.fabs(ad_r/(2*math.pi)*360)
                    
                else:
                    break
               
    neigh += [i for i in range(idx+1, r)]
    neigh += [i for i in range(l+1, idx)]
    neigh = sorted(neigh)

    return neigh


 

















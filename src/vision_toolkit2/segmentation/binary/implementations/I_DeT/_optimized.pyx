# distutils: language = c++

import numpy as np

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


def expand_cluster(double[:,:] g_npts, bool euclidean,
                    int n_s, int idx,
                    list neigh, double d_t,
                    int win_w, int min_pts, dict avlb):

    cdef list l_C_clus = [idx]
    cdef list n_neigh
    cdef int k = 0
    cdef int neigh_idx
    cdef int key

    avlb[idx] = False

    while k < len(neigh):
        neigh_idx = neigh[k]

        if avlb[neigh_idx] == True:
            avlb[neigh_idx] = False
            l_C_clus.append(neigh_idx)

            n_neigh = vareps_neighborhood(g_npts, euclidean,
                                          n_s, neigh_idx,
                                          d_t, win_w)

            if len(n_neigh) + 1 >= min_pts:
                for key in n_neigh:
                    if key not in neigh:
                        neigh.append(key)

        k += 1

    return l_C_clus, avlb


def vareps_neighborhood(double[:,:] g_npts,
                               bool euclidean,
                               int n_s, int idx,
                               double d_t, int win_w):

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
    cdef double dot, den, c


    cdef int r_stop = idx + win_w + 1
    if r_stop > n_s:
        r_stop = n_s

    cdef int l_stop = idx - win_w - 1
    if l_stop < 0:
        l_stop = 0


    if euclidean == True:

        with nogil:

            while r+1 < r_stop:
                if d_r < d_t:

                    r = r+1
                    d_r = math.sqrt((ref_g_npts[0]-g_npts[0,r])**2
                                    + (ref_g_npts[1]-g_npts[1,r])**2
                                    + (ref_g_npts[2]-g_npts[2,r])**2)
                else:
                    break

    else:

        with nogil:

            while r+1 < min(n_s, idx+win_w+1):
                if d_r < d_t:

                    r = r+1
                    n_2 = math.sqrt(g_npts[0,r]**
                                    + g_npts[1,r]**
                                    + g_npts[2,r]**2)

                    den = n_1 * n_2
                    if den <= 0.0:
                        d_r = 1e9
                    else:
                        dot = ref_g_npts[0]*g_npts[0,r] + ref_g_npts[1]*g_npts[1,r] + ref_g_npts[2]*g_npts[2,r]
                        c = dot / den
                        if c > 1.0:
                            c = 1.0
                        elif c < -1.0:
                            c = -1.0
                        ad_r = math.acos(c)
                        d_r = math.fabs(ad_r) * 180.0 / math.pi

                else:
                    break

    #to the left
    if euclidean == True:

        with nogil:

            while l > l_stop:

                if d_l < d_t:

                    l -= 1
                    d_l = math.sqrt((ref_g_npts[0]-g_npts[0,l])**2
                                    + (ref_g_npts[1]-g_npts[1,l])**2
                                    + (ref_g_npts[2]-g_npts[2,l])**2)
                else:
                    break

    else:

        with nogil:

            while l > max(0, idx-win_w-1):

                if d_l < d_t:

                    l -= 1
                    n_2 = math.sqrt(g_npts[0,l]**
                                    + g_npts[1,l]**
                                    + g_npts[2,l]**2)

                    den = n_1 * n_2
                    if den <= 0.0:
                        d_l = 1e9
                    else:
                        dot = (ref_g_npts[0]*g_npts[0,l]
                               + ref_g_npts[1]*g_npts[1,l]
                               + ref_g_npts[2]*g_npts[2,l])

                        c = dot / den
                        if c > 1.0:
                            c = 1.0
                        elif c < -1.0:
                            c = -1.0

                        ad_r = math.acos(c)
                        d_l = math.fabs(ad_r) * 180.0 / math.pi

                else:
                    break

    neigh += [i for i in range(idx+1, r)]
    neigh += [i for i in range(l+1, idx)]
    neigh = sorted(neigh)

    return neigh

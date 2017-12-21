'''
This benchmark is a comparison between scipy.linalg.svd and scipy.sparse.linalg.svds
Dung Tran: Dec/2017

In general, svd has better time performance than svds, however, require much more memory.
Reference: http://fa.bianp.net/blog/2012/singular-value-decomposition-in-scipy/
'''

import time
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.sparse import random

if __name__ == '__main__':

    dims = [10, 20, 30]

    svd_time = []
    svds_time = []

    for dim in dims:
        matrix_a = random(dim, dim, density=0.1, format='csc')
        k = min(matrix_a.shape) - 1

        # measure time for svds
        start = time.time()
        u, s, vt = svds(matrix_a, k)
        end = time.time()
        svds_time.append((dim, end - start))

        # measure time for svd
        start = time.time()
        u, s, vt = svd(matrix_a.todense())
        end = time.time()
        svd_time.append((dim, end - start))

    print "\nTiming performance:"
    print "\nsvd: {}".format(svd_time)
    print "\nsvds: {}".format(svds_time)


    # we decide to use svd to find a null space of a sparse matrix even it requires more memory

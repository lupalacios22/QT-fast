"""
Compare QT.py (original loop, copied unchanged) with QT_fast.py on a real RMSD
matrix, and check that both assign exactly the same clusters. Usage, from the
repository root:

    python tests/compare.py <matrix.npy> <N> <stride> <cutoff> <minsize> [--fast-only]

    <matrix.npy>  RMSD matrix saved by QT_fast.py (rmsd_matrix.npy)
    <N>, <stride> frames 0, stride, 2*stride, ... < N are used
"""
import os
import sys
import time
import numpy as np
import numpy.ma as ma

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import QT_fast


def qt_original(matrix, cutoff, minsize):
    # ---- copied from QT.py ------------------------------------------------
    N = matrix.shape[0]
    matrix[matrix > cutoff] = np.inf
    matrix[matrix == 0] = np.inf
    degrees = (matrix < np.inf).sum(axis=0)
    clusters_arr = np.ndarray(N, dtype=np.int64)
    clusters_arr.fill(-1)
    ncluster = 0
    while True:
        len_precluster = 0
        while True:
            biggest_node = degrees.argmax()
            precluster = []
            precluster.append(biggest_node)
            candidates = np.where(matrix[biggest_node] < np.inf)[0]
            next_ = biggest_node
            distances = matrix[next_][candidates]
            while True:
                next_ = candidates[distances.argmin()]
                precluster.append(next_)
                post_distances = matrix[next_][candidates]
                mask = post_distances > distances
                distances[mask] = post_distances[mask]
                if (distances == np.inf).all():
                    break
            degrees[biggest_node] = 0
            if len(precluster) > len_precluster:
                len_precluster = len(precluster)
                max_precluster = precluster
                max_node = biggest_node
                degrees = ma.masked_less(degrees, len_precluster)
            if not degrees.max():
                break
        if len(max_precluster) < minsize:
            break
        clusters_arr[max_precluster] = ncluster
        ncluster += 1
        print('>>> Cluster # {} found with {} frames at center {} <<<'.format(
              ncluster, len_precluster, max_node))
        matrix[max_precluster, :] = np.inf
        matrix[:, max_precluster] = np.inf
        degrees = (matrix < np.inf).sum(axis=0)
        if (degrees == 0).all():
            break
    return clusters_arr
    # -----------------------------------------------------------------------


path, N, stride, cutoff, minsize = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
fast_only = '--fast-only' in sys.argv
full = np.load(path)
sub = np.ascontiguousarray(full[:N:stride, :N:stride])
del full
print(f'== N={sub.shape[0]} (first {N} frames, stride {stride}), cutoff={cutoff}, minsize={minsize}')

t = time.time()
fast = QT_fast.qt_clustering(QT_fast.prepare_matrix(sub, cutoff), minsize)
t_fast = time.time() - t
print(f'   QT_fast.py: {t_fast:.1f} s, {fast.max() + 1} clusters')

if not fast_only:
    t = time.time()
    orig = qt_original(sub.copy(), cutoff, minsize)
    t_orig = time.time() - t
    print(f'   QT.py:      {t_orig:.1f} s, {orig.max() + 1} clusters')
    identical = np.array_equal(orig, fast)
    print(f'   IDENTICAL: {identical}   speed-up: x{t_orig / t_fast:.1f}')
    if not identical:
        print('   frames that differ:', int((orig != fast).sum()))
        sys.exit(1)

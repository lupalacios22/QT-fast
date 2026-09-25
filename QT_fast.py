#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
.. note ::

  | **Author      :** Roy Gonzalez Aleman
  | **Contact     :** [roy_gonzalez@fq.uh.cu, roy.gonzalez.aleman@gmail.com]
  | **Modified by :** L. Palacios, 2026-09-25

This file is a modified version of QT.py (https://github.com/rglez/QT),
distributed under the same license, the GNU GPL v3 (see LICENSE). If you use
it, please cite the original work:

  González-Alemán, R.; Hernández-Castillo, D.; Caballero, J.;
  Montero-Cabrera, L. A. Quality Threshold Clustering of Molecular Dynamics:
  A Word of Caution. J. Chem. Inf. Model. 2020, 60 (2), 467-472.
  https://doi.org/10.1021/acs.jcim.9b00558

Optimized version of QT.py. Same algorithm (Heyer et al.) and same clusters as
the original; only the way they are computed changes:

  - The RMSD matrix is saved as rmsd_matrix.npy (binary, float16) instead of
    a CSV file.
  - With -matrix, a precomputed matrix (.npy or .csv) is reused without
    reading the trajectory again, e.g. to change -cutoff or -minsize.
  - Only the selected atoms are loaded, not the whole system.
  - Each precluster grows with argmin + np.maximum in float32.
  - Between one cluster and the next, only the preclusters of centers that had
    a newly assigned frame among their neighbors are recomputed; the rest are
    unchanged and reused.
  - All outputs are written to -odir.

Examples:
  python QT_fast.py -traj traj.dcd -top system.psf -first 0 -last 10000 \\
      -sel "backbone and residue > 10" -cutoff 3 -minsize 100 -odir ./output
  python QT_fast.py -matrix ./output/rmsd_matrix.npy -cutoff 2.5 -minsize 100 \\
      -odir ./output_2.5
'''
import os
import sys
import time
import argparse
import numpy as np

# =============================================================================
# Useful functions
# =============================================================================


def parse_arguments():
    '''
    DESCRIPTION
    Parse all user arguments from the command line.

    Return:
        user_inputs (parser.argparse): namespace with user input arguments.
    '''

    # Initializing argparse ---------------------------------------------------
    desc = '\nQT: Implementation of the Quality Threshold Clustering algorithm by Heyer et. al.'
    parser = argparse.ArgumentParser(description=desc,
                                     add_help=True,
                                     epilog='As simple as that ;)')
    # Arguments: loading trajectory -------------------------------------------
    parser.add_argument('-top', dest='topology', action='store',
                        help='path to topology file (psf/pdb)',
                        type=str, required=False)
    parser.add_argument('-traj', dest='trajectory', action='store',
                        help='path to trajectory file',
                        type=str)
    parser.add_argument('-first', dest='first',  action='store',
                        help='first frame to analyze (starting from 0)',
                        type=int, required=False, default=0)
    parser.add_argument('-last', dest='last', action='store',
                        help='last frame to analyze (starting from 0)',
                        type=int, required=False, default=-1)
    parser.add_argument('-stride', dest='stride', action='store',
                        help='stride of frames to analyze',
                        type=int, required=False, default=1)
    parser.add_argument('-sel', dest='selection', action='store',
                        help='atom selection (MDTraj syntax)',
                        type=str, required=False, default='all')
    # Arguments: reusing a matrix ---------------------------------------------
    parser.add_argument('-matrix', dest='matrix', action='store',
                        help='precomputed RMSD matrix (.npy or .csv); '
                             'if given, -traj/-top/-sel/-first/-last/-stride are ignored',
                        type=str, required=False)
    # Arguments: clustering ---------------------------------------------------
    parser.add_argument('-cutoff', action='store', dest='cutoff',
                        help='RMSD cutoff for pairwise comparisons in A',
                        type=float, required=False, default=1.0)
    parser.add_argument('-minsize', action='store', dest='minsize',
                        help='minimum number of frames inside returned clusters',
                        type=int, required=False, default=2)
    # Arguments: analysis -----------------------------------------------------
    parser.add_argument('-odir', action='store', dest='outdir',
                        help='output directory to store analysis',
                        type=str, required=False, default='./')
    user_inputs = parser.parse_args()
    return user_inputs


def load_trajectory(args):
    '''
    DESCRIPTION
    Loads trajectory file using MDTraj. If trajectory format is h5, lh5 or
    pdb, topology file is not required. Otherwise, you should specify a
    topology file. Only the selected atoms are read into memory.

    Arguments:
        args (argparse.Namespace): user input parameters parsed by argparse.
    Return:
        trajectory (mdtraj.Trajectory): trajectory object for further analysis.
    '''
    import mdtraj as md

    traj_file = args.trajectory
    traj_ext = traj_file.split('.')[-1]
    # Does trajectory file format need topology ? -----------------------------
    if traj_ext in ['h5', 'lh5', 'pdb']:
        topology = md.load_topology(traj_file)
        top_arg = None
    else:
        topology = md.load_topology(args.topology)
        top_arg = args.topology

    # Reduce RAM consumption by loading selected atoms only -------------------
    sel_indx = None
    if args.selection != 'all':
        try:
            sel_indx = topology.select(args.selection)
        except ValueError:
            print('Specified selection is invalid')
            sys.exit()
        if sel_indx.size == 0:
            print('Specified selection in your system corresponds to no atoms')
            sys.exit()
    if top_arg is None:
        trajectory = md.load(traj_file, atom_indices=sel_indx)
    else:
        trajectory = md.load(traj_file, top=top_arg, atom_indices=sel_indx)
    trajectory = trajectory[args.first:args.last:args.stride]

    # Center coordinates of loaded trajectory ---------------------------------
    trajectory.center_coordinates()
    return trajectory


def compute_rmsd_matrix(trajectory):
    '''
    DESCRIPTION
    Pairwise RMSD matrix (in A), stored as float16 exactly like QT.py.
    '''
    import mdtraj as md

    N = trajectory.n_frames
    matrix = np.ndarray((N, N), dtype=np.float16)
    for i in range(N):
        matrix[i] = md.rmsd(trajectory, trajectory, i, precentered=True)*10
    return matrix


def load_matrix(path, outdir):
    '''
    DESCRIPTION
    Loads a precomputed RMSD matrix. A .csv (as written by QT.py) is converted
    to float16, which reproduces the original in-memory matrix exactly, and
    cached as rmsd_matrix.npy inside outdir for the next runs.
    '''
    if path.endswith('.npy'):
        return np.load(path).astype(np.float16, copy=False)
    try:
        import pandas as pd
        matrix = pd.read_csv(path, header=None, dtype=np.float64).values
    except ImportError:
        matrix = np.loadtxt(path, delimiter=',', dtype=np.float64)
    matrix = matrix.astype(np.float16)
    np.save(os.path.join(outdir, 'rmsd_matrix.npy'), matrix)
    print(">>> RMSD matrix converted and saved as 'rmsd_matrix.npy' <<<")
    return matrix


# =============================================================================
# QT algorithm
# =============================================================================


def prepare_matrix(matrix, cutoff):
    '''
    DESCRIPTION
    Same masking as QT.py (x > cutoff and x == 0 become inf), returned as a new
    float32 matrix. The comparison is done at float16 precision, as in QT.py,
    and float16 -> float32 is exact, so every comparison gives the same answer.
    '''
    work = matrix.astype(np.float32)
    work[work > np.float32(np.float16(cutoff))] = np.inf
    work[work == 0] = np.inf
    return work


def grow_precluster(work, center, members=None):
    '''
    DESCRIPTION
    Greedy growth of the precluster around `center`: at each step add the
    candidate whose largest distance to the current members is smallest, until
    no candidate is within the cutoff of every member. Returns its size; if a
    list is passed as `members`, it is filled with the frames in the order
    QT.py adds them.
    '''
    candidates = np.flatnonzero(work[center] < np.inf)
    if members is not None:
        members.append(center)
    if candidates.size == 0:
        return 1
    distances = work[center, candidates]
    size = 1
    while True:
        k = distances.argmin()
        if distances[k] == np.inf:
            break
        next_ = candidates[k]
        size += 1
        if members is not None:
            members.append(next_)
        np.maximum(distances, work[next_, candidates], out=distances)
    return size


def qt_clustering(work, minsize):
    '''
    DESCRIPTION
    Quality Threshold clustering on a matrix prepared with prepare_matrix().
    `work` is modified in place.

    Each cluster is the largest precluster over all candidate centers, visited
    in order of decreasing degree (ties by frame index) and skipping those
    whose degree is below the best size found so far, exactly as QT.py does.
    Precluster sizes are cached between clusters and recomputed only for the
    centers that had a removed frame among their neighbours.

    Return:
        clusters_arr (np.ndarray): cluster id per frame (-1 = unassigned).
    '''
    N = work.shape[0]
    degrees = (work < np.inf).sum(axis=0)
    cached_size = np.full(N, -1, dtype=np.int64)
    clusters_arr = np.full(N, -1, dtype=np.int64)
    frame_index = np.arange(N)

    ncluster = 0
    while degrees.any():
        order = np.lexsort((frame_index, -degrees))
        len_precluster = 0
        max_node = -1
        for node in order:
            if degrees[node] < len_precluster or degrees[node] == 0:
                break
            if cached_size[node] < 0:
                cached_size[node] = grow_precluster(work, node)
            if cached_size[node] > len_precluster:
                len_precluster = cached_size[node]
                max_node = node
        # General break if minsize is reached ---------------------------------
        if len_precluster < minsize:
            break

        # ---- Store cluster frames -------------------------------------------
        max_precluster = []
        grow_precluster(work, max_node, max_precluster)
        max_precluster = np.array(max_precluster)
        clusters_arr[max_precluster] = ncluster
        ncluster += 1
        print('>>> Cluster # {} found with {} frames at center {} <<<'.format(
              ncluster, len_precluster, max_node), flush=True)

        # ---- Update matrix & degrees (discard found clusters) ---------------
        affected = (work[:, max_precluster] < np.inf).any(axis=1)
        cached_size[affected] = -1
        degrees -= (work[max_precluster, :] < np.inf).sum(axis=0)
        work[max_precluster, :] = np.inf
        work[:, max_precluster] = np.inf
        degrees[max_precluster] = 0

    return clusters_arr


def write_results(clusters_arr, outdir):
    '''
    DESCRIPTION
    Same output files as QT.py.
    '''
    # simple format
    np.savetxt(os.path.join(outdir, 'QT_Clusters.txt'), clusters_arr, fmt='%i')

    # NMRcluster format. VMD interface
    with open(os.path.join(outdir, 'QT_Visualization.log'), 'wt') as clq:
        for numcluster in np.unique(clusters_arr):
            clq.write('{}:\n'.format(numcluster))
            members = ' '.join([str(x + 1)
                                for x in np.where(clusters_arr == numcluster)[0]])
            clq.write('Members: ' + members + '\n\n')


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    inputs = parse_arguments()
    os.makedirs(inputs.outdir, exist_ok=True)
    t0 = time.time()

    # ---- Get the RMSD matrix ------------------------------------------------
    if inputs.matrix:
        matrix = load_matrix(inputs.matrix, inputs.outdir)
        print('>>> RMSD matrix loaded from {} <<<'.format(inputs.matrix))
    else:
        sms = '\n\n ATTENTION !!! No trajectory passed.Run with -h for help.'
        assert inputs.trajectory, sms
        trajectory = load_trajectory(inputs)
        matrix = compute_rmsd_matrix(trajectory)
        print('>>> Calculation of the RMSD matrix completed <<<')
        np.save(os.path.join(inputs.outdir, 'rmsd_matrix.npy'), matrix)
        print(">>> RMSD matrix saved as 'rmsd_matrix.npy' <<<")
    print('    N = {} frames ({:.0f} s)'.format(matrix.shape[0], time.time() - t0))

    # ---- QT -----------------------------------------------------------------
    t1 = time.time()
    work = prepare_matrix(matrix, inputs.cutoff)
    del matrix
    clusters_arr = qt_clustering(work, inputs.minsize)
    print('>>> QT completed in {:.0f} s <<<'.format(time.time() - t1))

    write_results(clusters_arr, inputs.outdir)
    print('>>> Results written to {} <<<'.format(inputs.outdir))

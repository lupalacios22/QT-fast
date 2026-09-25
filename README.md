# QT-fast

A faster version of [**QT**](https://github.com/rglez/QT), Roy González-Alemán's
implementation of the *Quality Threshold* clustering algorithm (Heyer et al.,
1999) for molecular dynamics trajectories.

`QT_fast.py` runs **the same algorithm and returns exactly the same clusters**
as `QT.py`, but:

- the clustering step is **~5 times faster**;
- the RMSD matrix is saved in binary format (`.npy`);
- the matrix can be **reused** to try a different `-cutoff` or `-minsize`
  without reading the trajectory again.

## Credit and citation

The algorithm implementation and the original code are by
**Roy González-Alemán** ([rglez/QT](https://github.com/rglez/QT)). The `QT.py`
file in this repository is the original, unmodified. If you use this code,
please cite the original work:

> González-Alemán, R.; Hernández-Castillo, D.; Caballero, J.;
> Montero-Cabrera, L. A. *Quality Threshold Clustering of Molecular Dynamics:
> A Word of Caution.* J. Chem. Inf. Model. **2020**, 60 (2), 467–472.
> https://doi.org/10.1021/acs.jcim.9b00558

> Heyer, L. J.; Kruglyak, S.; Yooseph, S. *Exploring Expression Data:
> Identification and Analysis of Coexpressed Genes.* Genome Res. **1999**,
> 9 (11), 1106–1115. https://doi.org/10.1101/gr.9.11.1106

## Contents

| File | Description |
|---|---|
| `QT.py` | Original implementation from [rglez/QT](https://github.com/rglez/QT), unmodified |
| `QT_fast.py` | Optimized version (this work) |
| `tests/compare.py` | Runs both versions on the same matrix and checks that the results are identical |
| `LICENSE` | GNU GPL v3, the same license as the original |

## Usage
  **Requirements:** Python 3, NumPy and MDTraj (MDTraj is only needed when reading a trajectory; pandas is optional, to read CSV matrices faster).

The arguments are the same as in `QT.py`, plus `-matrix`.

**1. From a trajectory.** Computes the RMSD matrix, saves it as
`rmsd_matrix.npy`, and runs the clustering:

```bash
python QT_fast.py -traj traj.dcd -top system.psf -first 0 -last 10000 \
    -sel "backbone and residue > 10" -cutoff 3 -minsize 100 -odir ./cut3
```

**2. From a precomputed matrix.** Changes `-cutoff` or `-minsize` without
recomputing anything:

```bash
python QT_fast.py -matrix ./cut3/rmsd_matrix.npy -cutoff 2.5 -minsize 100 -odir ./cut2.5
```

`-matrix` also accepts an `rmsd_matrix.csv` containing the full matrix. On
first use, the CSV is converted and saved as `rmsd_matrix.npy` in `-odir` for
later runs.

The matrix depends on the trajectory, the atom selection (`-sel`), and the
frames used (`-first`, `-last`, `-stride`). If any of these change, the matrix
must be recomputed (mode 1).

| Argument | Description | Default |
|---|---|---|
| `-traj` | Trajectory (any format MDTraj can read) | — |
| `-top` | Topology (psf/pdb) | — |
| `-first`, `-last`, `-stride` | Frames to analyze (`last` is excluded, as in Python) | `0`, `-1`, `1` |
| `-sel` | Atom selection (MDTraj syntax) | `all` |
| `-matrix` | Precomputed matrix (`.npy` or `.csv`); replaces the five arguments above | — |
| `-cutoff` | RMSD cutoff in Å | `1.0` |
| `-minsize` | Minimum cluster size | `2` |
| `-odir` | Output directory (created if it does not exist) | `./` |

### Outputs (in `-odir`)

- `rmsd_matrix.npy`: RMSD matrix in Å (float16, N × N).
- `QT_Clusters.txt`: one cluster ID per frame; `-1` means unassigned.
  Same format as `QT.py`.
- `QT_Visualization.log`: members of each cluster, in the format read by VMD.
  Same format as `QT.py`.

## What changes with respect to `QT.py`

The algorithm is the same, step by step. Centers are tried in the same order
(decreasing degree, with ties broken by lowest frame index) and with the same
pruning, and each precluster grows by adding the same frames in the same order.
Only the way it is computed changes:

1. **Precluster growth.** Each step is reduced to one `argmin` and one
   `np.maximum` over the candidate distances, instead of the ~7 numpy
   operations in the original. Computations are done in float32 instead of
   float16: numpy emulates float16 in software, and the float16 → float32
   conversion is exact, so every comparison gives the same result.

2. **Caching between clusters.** After each cluster is found, `QT.py`
   recomputes the precluster of every candidate center from scratch. However,
   a center's precluster depends only on its neighbors: if none of them was
   part of the cluster just removed, its precluster is the same as before.
   `QT_fast.py` stores the size of each precluster and recomputes only those
   of the affected centers. Degrees are also updated by subtraction instead of
   being recounted over the whole matrix.

3. **Input and output.**
   - The matrix is saved as `.npy`: 200 MB for N = 10 000, loaded in under a
     second.
   - `-matrix` reuses a matrix without touching the trajectory.
   - Only the atoms in `-sel` are loaded from the trajectory, instead of the
     whole system.
   - `-odir` is honored (in `QT.py`, outputs are always written to the current
     directory).

## Validation and performance

Both versions were compared on RMSD matrices from TAMD simulations of a peptide,
on a 6-core machine. In every case, the cluster assignment was **identical
frame by frame**.

| Frames | Cutoff (Å) | minsize | `QT.py` | `QT_fast.py` | Speed-up |
|---:|---:|---:|---:|---:|---:|
| 2 000 | 2.0 | 20 | 8.2 s | 1.7 s | ×4.8 |
| 2 000 | 2.5 | 20 | 13.4 s | 3.6 s | ×3.7 |
| 2 000 | 3.0 | 20 | 23.7 s | 6.8 s | ×3.5 |
| 5 000 | 3.0 | 50 | 159 s | 35 s | ×4.5 |
| 10 000 | 3.0 | 100 | ~18 min | 3.3 min | ×5.5 |

In the 10 000-frame run, the `QT_Clusters.txt` and `QT_Visualization.log` files
were also byte-for-byte identical to those produced by `QT.py`.

Timings refer to the clustering step only. Computing the RMSD matrix takes the
same time in both versions (here, ~1.5 min for 10 000 frames). The cost of QT
grows quickly with N and with the cutoff.

To repeat the comparison with your own data:

```bash
python tests/compare.py ./cut3/rmsd_matrix.npy 2000 1 3.0 20
```

## Notes

- **Memory.** About 6 bytes × N² are needed: the float16 matrix plus the float32
  working copy. That is ~600 MB for N = 10 000 and ~5.4 GB for N = 30 000.
- **Identical frames.** As in `QT.py`, pairs with an RMSD of exactly 0 are
  treated as non-neighbors (the line `matrix[matrix == 0] = np.inf` removes the
  diagonal, but also those pairs). This behavior was kept to give the same
  results. If your trajectory contains repeated frames, for example from
  concatenating overlapping runs, remove them first.
- **Degenerate case.** If a center has no neighbors within the cutoff, `QT.py`
  stops with an error; `QT_fast.py` treats it as a precluster of size 1.

## License

GNU General Public License v3.0, the same as the original project. See
[`LICENSE`](LICENSE).

# Subtomogram averaging of the simulated CZII proteins (RELION 4.0)

STA of the 7 globular proteins in the simulated tomograms of `train_dir_6`
(config `czii_c6.toml`, 50 tomograms, 10 A/voxel), using **ground-truth
orientations** translated into RELION's convention. Two tomogram variants:

- **basic** = plain Gaussian-noise tomograms
- **faket** = FakET style-transfer tomograms

## Code vs data — where things live

**Code (here):** `/user/muth9/u12095/czii-protein-challenge/STA/`
| file | purpose |
|------|---------|
| `extract_particles.py` | cut out subtomograms + write RELION-4 `particles.star` |
| `relion_euler.py`      | RELION euler<->matrix (verified port of src/euler.cpp) |
| `env.sh`               | sets PATH to RELION + OpenMPI (source before GUI/refine) |
| `run_gt_average.sh`    | STA average at GT angles, one species |
| `run_refine3d.sh`      | de-novo 3D auto-refine (GPU+MPI), one species |
| `run_all.sh`           | loop all 7 species |
| `RUN_INSTRUCTIONS.md`  | how to produce the volumes (terminal + GUI) |

**Data / outputs (stay in the dataset dir):**
`/mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_dataset_20260423/`
- `relion_sta/<species>/`        (basic variant)
- `relion_sta_faket/<species>/`  (faket variant)

each containing `particles.star`, `Subtomograms/`, `wedge_ctf.mrc`,
`initial_ref.mrc`, and results in `GTaverage/` (and `Refine3D/` if refined).

RELION 4.0.2 binary: `/user/muth9/u12095/software/relion/install/bin`
(not in the `pro-revelio` env, which only has python/mrcfile/scipy).

## Run

```bash
cd /user/muth9/u12095/czii-protein-challenge/STA

# (re)extract subtomos + star files  (writes into the dataset dir)
python extract_particles.py --species all --tomos all --variant basic
python extract_particles.py --species all --tomos all --variant faket

# STA averages at GT orientations (fast, CPU)
bash run_all.sh gt basic
bash run_all.sh gt faket

# de-novo auto-refine (GPU+MPI, gives FSC)   -- see RUN_INSTRUCTIONS.md
bash run_all.sh refine basic
```
Single species, e.g.: `bash run_gt_average.sh faket ribosome`.
The averaged volume is `<...>/<species>/GTaverage/run_it001_class001.mrc`.

## ⚠️ The overlay JSON orientations are BUGGY — do not use them

`train_dir_6/overlay/*/<species>.json` `transformation_` matrices are wrong:
`faket_polnet/utils/label_transform.py` reads PolNet's scalar-first quaternion
`(w,x,y,z)` as scalar-last `(x,y,z,w)`, scrambling every orientation (proof:
aligning clean densities gives ~random consistency with the overlay matrix vs
0.90-0.98 with the correct one). `extract_particles.py` therefore reads the
**motif-list CSV** (`simulation_dir_6/motif_lists/tomo_motif_list_*.csv`, Q1..Q4)
and rebuilds the matrix correctly. Positions in the overlay are fine; we use the
CSV for both.

## Validated GT -> RELION recipe (per particle)

```
R_pn   = quaternion_matrix(w=Q1, x=Q2, y=Q3, z=Q4)     # scalar-first
R_tomo = M @ R_pn @ M                                   # M=diag(1,1,-1): IMOD z-flip
rot,tilt,psi = relion_euler.matrix2angles(R_tomo)       # RELION's exact ZYZ convention
coords : x/10 , y/10 , (nz-1) - z/10                    # X,Y direct; Z inverted
```
Verified via: coordinate sweep (Z-flip), source trace + clean-density alignment
(scalar-first), RELION clean reconstruction (direction), z-mirror test (M R_pn M),
and euler roundtrip vs RELION source (1e-16).

## Data facts / caveats

- pixel size 10 A; box 630x630x184; subtomo box 64.
- tomograms have a **missing wedge** (+/-60 deg, tilt axis Y) but **no CTF**;
  `wedge_ctf.mrc` is a pure missing-wedge mask (RELION requires a `rlnCtfImage`).
- protein is dark in the raw data; extraction inverts to white protein.
- low orientational contrast at 10 A -> expect overall-shape recovery, limited
  high-res detail; reconstructions are **Z-mirrored** vs the true PDB.

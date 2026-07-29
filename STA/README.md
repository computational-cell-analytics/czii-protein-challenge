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

RELION comes from the cluster module (`module load gcc/13.2.0 openmpi/5.0.7
relion/4.0.1`, GPU-enabled; `relion/5.0.0` also available) -- just `source env.sh`.
It is not in the `pro-revelio` env, which only has python/mrcfile/scipy.

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
R_pn = quaternion_matrix(w=Q1, x=Q2, y=Q3, z=Q4)       # scalar-first
rot,tilt,psi = relion_euler.matrix2angles(R_pn)         # RELION's exact ZYZ convention
coords : x/10 , y/10 , z/10                             # NO flip
```
**NO z-flip and NO mirror** — the IMOD reconstruction preserves the coordinate
frame. Verified via: global correlation clean-vs-basic tomogram (identity beats
z-flip 9x); a SOLID particle (ribosome) sits at iz=z/10 in both clean and basic
(contrast ~2-4 vs ~0.5 for flip); source trace + clean-density alignment
(scalar-first quaternion); RELION clean reconstruction (direction = R_pn, no
transpose); euler roundtrip vs RELION source (1e-16).

> Correction note: an earlier version used a z-flip + `M R_pn M` mirror. That was
> wrong — it came from a coordinate check on the **VLP**, whose hollow shell fools
> a centre-vs-edge metric. The corrected recipe gives a ~38x sharper ribosome
> average. Always verify coordinate conventions with a SOLID particle.

## Data facts / caveats

- pixel size 10 A; box 630x630x184; subtomo box 64.
- tomograms have a **missing wedge** (+/-60 deg, tilt axis Y) but **no CTF**;
  `wedge_ctf.mrc` is a pure missing-wedge mask (RELION requires a `rlnCtfImage`).
- protein is dark in the raw data; extraction inverts to white protein.
- low orientational contrast at 10 A -> expect overall-shape recovery, limited
  high-res detail. Handedness is preserved (no flip), so maps should match true
  PDB chirality.

## Experimental data (real CZII) — de-novo STA

Real CZII tomograms (OME-Zarr) + copick pick locations, orientations UNKNOWN so
RELION finds them by de-novo 3D auto-refine, with a low-pass-filtered PDB model as
the initial reference.

- data:  `/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/tomograms_VoxelSpacing10/` (zarr, 10.012 A)
- picks: `.../ground_truth/structure_for_detection/tomograms_VoxelSpacing10/<run>/Picks/<species>.json`
- project (outputs): `.../czii_challenge/relion_sta_experimental/<species>/`
- 6 species (no albumin); conventions verified on the real data: NO z-flip, protein dark -> inverted.
- reference = sim's PDB-derived 10 A templates, padded to box 64 + low-pass 50 A
  (unbiased starting model), built by `build_reference.py`.

```bash
micromamba activate pro-revelio
cd /user/muth9/u12095/czii-protein-challenge/STA
python extract_experimental.py --species all --runs all   # zarr+copick -> subtomos + star
python build_reference.py                                  # lowpass PDB initial_ref.mrc per species
# then, on the GPU desktop:
source env.sh
bash run_all.sh refine experimental                        # de-novo auto-refine (GPU+MPI)
#   -> <species>/Refine3D/run_class001.mrc  (+ half-maps, resolution in run_model.star)
```
particle counts: ribosome 5636, apo-ferritin 4521, thyroglobulin 1210,
beta-galactosidase 670, beta-amylase 570, virus-like-particle 551.

For an apples-to-apples sim-vs-experiment comparison, give the SIM runs the same
PDB reference and de-novo-refine them too:
```bash
python build_reference.py --project <dataset>/relion_sta
python build_reference.py --project <dataset>/relion_sta_faket
bash run_all.sh refine basic ; bash run_all.sh refine faket
```
Then compare each variant's refined map to the same low-pass PDB (map-to-model
FSC) so basic / faket / experimental sit on one common yardstick.

Caveat: the missing-wedge model for the real data is approximated as +/-60 deg /
tilt-axis Y (same as the sim); the true per-tomogram geometry isn't used (we only
have reconstructed volumes, not tilt series).

# How to get the STA 3D volumes

Code lives here (`/user/muth9/u12095/czii-protein-challenge/STA/`); outputs are
written into the dataset dir
(`/mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_dataset_20260423/relion_sta[_faket]/`).
Run everything **in a terminal on your Jupyter desktop** (node agq004: 3x RTX5000
GPUs + 24 CPUs) — nothing needs to be submitted to a queue.

Two kinds of result:
- **A. GT-angle average** (recommended here): reconstruct at the validated
  ground-truth angles. Fast, CPU-only. `-> <species>/GTaverage/run_it001_class001.mrc`
- **B. De-novo 3D auto-refine**: RELION searches orientations itself, gives a
  gold-standard FSC. GPU + MPI. `-> <species>/Refine3D/run_class001.mrc`

RELION comes from the cluster module: `module load gcc/13.2.0 openmpi/5.0.7
relion/4.0.1` (GPU-enabled; `relion/5.0.0` also available). Easiest: `source env.sh`.

---

## Option 1 — Terminal (simplest)

```bash
cd /user/muth9/u12095/czii-protein-challenge/STA

# A) GT-angle averages (all 7 species)
bash run_all.sh gt basic       # plain-noise tomograms
bash run_all.sh gt faket       # style-transfer tomograms

# B) de-novo auto-refine (GPU)
bash run_all.sh refine basic
```
One species: `bash run_gt_average.sh faket ribosome` or `bash run_refine3d.sh basic ribosome`.
View: `source env.sh; relion_display --i <dataset>/relion_sta/ribosome/GTaverage/run_it001_class001.mrc`
(or open the .mrc in ChimeraX / 3dmod).

---

## Option 2 — RELION 4 GUI (interactive; gives result type B)

The GUI's 3D auto-refine always *searches* orientations (type B). For the
GT-angle average, use Option 1.

1. Environment:
   ```bash
   source /user/muth9/u12095/czii-protein-challenge/STA/env.sh
   ```
2. The RELION project must be the **species folder** (star paths are relative to
   it). Launch the GUI there:
   ```bash
   cd /mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_dataset_20260423/relion_sta/ribosome
   relion &
   ```
   Confirm "set up a new RELION project here".
3. Job type **3D auto-refine**:
   - **I/O**: Input images STAR = `particles.star`; Reference map = `initial_ref.mrc`;
     Reference mask = (empty).
   - **Reference**: greyscale? **No**; Initial low-pass = **50** A; Symmetry = **C1**.
   - **CTF**: Do CTF-correction? **Yes** (uses the 3D wedge in the star).
   - **Optimisation**: Mask diameter (A) per species: vlp 320, ribosome 300,
     thyroglobulin 250, beta-galactosidase 180, apo-ferritin 130, beta-amylase 130,
     albumin 100. Flatten & enforce non-negative solvent? Yes.
   - **Compute**: Use GPU? **Yes**; Which GPUs = (blank = auto); Pre-read into RAM? Yes.
   - **Running**: MPI procs = **3**; threads = **6**; Submit to queue? **No**.
   - **Run!**
4. Result: `Refine3D/jobNNN/run_class001.mrc` (+ half maps, FSC in `run_model.star`).
5. Repeat per species: launch `relion` in that species' folder; set mask diameter.

---

## Notes

- `relion_refine` (serial) is CPU-only; the GPU build is `relion_refine_mpi`
  (used automatically when MPI procs > 1 / in `run_refine3d.sh`).
- The reconstruction preserves the coordinate frame (no z-flip / no mirror), so
  maps should have the correct handedness vs a deposited structure.
- 10 A/voxel, SNR ~0.11, smooth simulated densities -> expect low resolution,
  mostly overall-shape recovery.

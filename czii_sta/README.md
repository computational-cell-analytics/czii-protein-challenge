# CZII ribosome STA (RELION 4.0.1)

De-novo subtomogram averaging of CZII ribosomes, built by applying the RELION-4 STA
tutorial workflow. Independent of the older `STA/` directory — nothing was carried
over from it.

- **scripts** (here): `/user/muth9/u12095/czii-protein-challenge/czii_sta/`
- **data**: `/mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_sta_ribosome/`

## Conventions — all measured, none assumed

| | value | how |
|---|---|---|
| pixel size | 10.012 Å | zarr `.zattrs` |
| tomogram | 184 × 630 × 630 | zarr level 0 |
| contrast | protein **dark** → inverted on extraction | density at picks is 2–4 σ below the global mean |
| coordinates | Å ÷ 10.012 | picks JSON `"unit": "angstrom"` |
| orientations | **none** — all 20367 matrices are the identity | → de novo, no GT-angle shortcut |
| missing wedge | **±45° about Y** | portal `tilt_range` 90°, uniform across runs; confirmed by where the power spectrum falls off (kX–kZ flat to ~45°, kY–kZ drops immediately) |
| symmetry | C1 | a ribosome is asymmetric |
| Nyquist | **20 Å** | 10.012 Å/voxel |

The tomograms are already CTF-corrected (`ctf_corrected=True` on the portal), so no
CTF correction or CTF refinement is needed or possible here.

## Pipeline

| # | script | partition | produces |
|---|---|---|---|
| 1 | `run_sbatch_extract.sbatch` | CPU | `Subtomograms/`, `particles.star`, `wedge_ctf.mrc` |
| 2 | `run_sbatch_initialmodel.sbatch` | GPU | `InitialModel/run_it100_class001.mrc` |
| 3 | `run_sbatch_refine3d.sbatch` | GPU | `Refine3D/run_class001.mrc` + half-maps |
| 4 | `run_sbatch_mask.sbatch` | CPU | `Mask/mask.mrc` (threshold picked automatically) |
| 5 | `run_sbatch_postprocess.sbatch` | CPU | `PostProcess/postprocess.mrc` + corrected FSC |

Submit the whole chain with `bash run_all.sh` — it wires the five jobs together with
`--dependency=afterok`, so a failure stops the rest rather than feeding bad inputs
downstream. `bash run_all.sh 3` restarts from step 3. Or submit each `sbatch`
individually.

## Current settings

Box 64 (641 Å), `--particle_diameter 300`, `--fraction 0.15` (≈1021 of 6809 particles,
seed 0, drawn across all runs). The fraction is a **smoke test** to shake out the
pipeline cheaply — rerun with `--fraction 1.0` for the real result. Expect ~2.6×
worse signal at 15%.

## Things carried over from the tutorial that are easy to get wrong

- `--zero_mask` is **mandatory** with `--gpu` + 3D input; without it RELION dies with
  "Noise-masking not supported with acceleration and 3D input".
- `--healpix_order` / `--auto_local_healpix_order` are stated explicitly because the
  RELION **GUI** defaults (7.5°/1.8°) differ from the **CLI** defaults (15°/3.75°).
  Omitting them silently gives a coarser search.
- `--ctf` must actually be passed, or `rlnCtfImage` is ignored and the missing wedge
  never applied.
- GPU and CPU nodes need **different module incantations** (`env_gpu.sh` vs
  `env_cpu.sh`); the wrong one leaves relion off PATH with no error.
- Refinement wall time is dominated by whether the global-search phase happens: it
  does when the initial sampling is coarser than `--auto_local_healpix_order`.

## The mask threshold

`pick_mask_threshold.py` sets it by Otsu on the refined map's voxel histogram — the
map is bimodal (solvent vs particle), so the threshold minimising within-class
variance separates them without any external knowledge of the specimen.

It is a **heuristic**, so it reports rather than just decides: enclosed volume, the
equivalent-sphere diameter, and mean+1..4σ for comparison. If the enclosed volume
implies a diameter outside 0.5–1.6× the expected 300 Å it prints a loud warning —
treat that as "look at the map before trusting the mask".

Override at any time:

```bash
MASK_THRESHOLD=0.012 bash run_all.sh 4
```

`--extend_inimask 3` (30 Å) and `--width_soft_edge 6` (60 Å) are still unvalidated
choices. A hard-edged or over-tight mask inflates the FSC; the corrected curve from
postprocess is what tells you whether it did.

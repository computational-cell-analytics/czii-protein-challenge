#!/usr/bin/env python
"""
Extract subtomograms + write RELION-4 particle STAR files for the EXPERIMENTAL
CZII data (OME-Zarr tomograms + copick pick JSONs). Orientations are UNKNOWN, so
angles are set to 0 and RELION finds them by de-novo 3D auto-refine.

Verified conventions (real data): coords map directly (NO z-flip); protein is
dark -> inverted to white. Pixel size 10.012 A. Box 64.

Run:
  micromamba activate pro-revelio
  python extract_experimental.py --species all --runs all      # everything
  python extract_experimental.py --species ribosome --n-runs 5 # quick test
"""
import argparse, glob, json, os
import numpy as np, mrcfile, zarr

CZII = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge"   # raw data (lustre-grete only)
DATA = f"{CZII}/data/tomograms_VoxelSpacing10"
PICKS = f"{CZII}/ground_truth/structure_for_detection/tomograms_VoxelSpacing10"
# outputs on vast-nhr so the Intel GPU nodes / Jupyter desktop can read them
ROOT = "/mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_dataset_20260423/relion_sta_experimental"
APIX = 10.012
SPECIES_ALL = ["ribosome", "virus-like-particle", "beta-galactosidase",
               "thyroglobulin", "apo-ferritin", "beta-amylase"]

def zarr_vol(run):
    g = glob.glob(f"{DATA}/{run}/*.zarr")
    if not g: return None
    return np.array(zarr.open(g[0], mode="r")["0"]).astype(np.float32)   # level 0 = full res

def picks(run, sp):
    f = f"{PICKS}/{run}/Picks/{sp}.json"
    if not os.path.exists(f): return []
    return [(p["location"]["x"], p["location"]["y"], p["location"]["z"])
            for p in json.load(open(f))["points"]]

def write_wedge(path, box, tilt_max=60.0):
    c = box // 2; k, i, j = np.mgrid[0:box, 0:box, 0:box]
    mask = (np.degrees(np.arctan2(np.abs(k-c), np.abs(j-c))) <= tilt_max).astype(np.float32)
    with mrcfile.new(path, overwrite=True) as m: m.set_data(mask); m.voxel_size = APIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="+", default=["all"])
    ap.add_argument("--runs", nargs="+", default=["all"])
    ap.add_argument("--n-runs", type=int, default=0, help="limit to first N runs (0=all)")
    ap.add_argument("--box", type=int, default=64)
    ap.add_argument("--no-invert", action="store_true")
    args = ap.parse_args()

    species = SPECIES_ALL if args.species == ["all"] else args.species
    runs = sorted(os.path.basename(p) for p in glob.glob(f"{DATA}/*") if os.path.isdir(p)) \
        if args.runs == ["all"] else args.runs
    if args.n_runs: runs = runs[:args.n_runs]
    box, H = args.box, args.box // 2
    invert = not args.no_invert

    rows = {s: [] for s in species}
    refs = {s: (np.zeros((box,box,box), np.float64), 0) for s in species}
    for s in species:
        os.makedirs(f"{ROOT}/{s}/Subtomograms", exist_ok=True)
        write_wedge(f"{ROOT}/{s}/wedge_ctf.mrc", box)

    for gi, run in enumerate(runs):
        vol = zarr_vol(run)
        if vol is None:
            print(f"[run {run}] no zarr, skip"); continue
        nz, ny, nx = vol.shape
        for s in species:
            pk = picks(run, s)
            if not pk: continue
            os.makedirs(f"{ROOT}/{s}/Subtomograms/{run}", exist_ok=True)
            kept = 0
            for j, (X, Y, Z) in enumerate(pk):
                ix, iy, iz = int(round(X/APIX)), int(round(Y/APIX)), int(round(Z/APIX))  # NO flip
                if ix-H<0 or ix+H>nx or iy-H<0 or iy+H>ny or iz-H<0 or iz+H>nz: continue
                b = vol[iz-H:iz+H, iy-H:iy+H, ix-H:ix+H].astype(np.float32)
                b = (b - b.mean())/(b.std()+1e-9)
                if invert: b = -b                                  # protein dark -> white
                rel = f"Subtomograms/{run}/{s}_{run}_{j:04d}.mrc"
                with mrcfile.new(f"{ROOT}/{s}/{rel}", overwrite=True) as om:
                    om.set_data(b); om.voxel_size = APIX
                rows[s].append((rel, "wedge_ctf.mrc", g_zarr(run), ix, iy, iz, gi+1, len(rows[s])%2+1))
                acc, n = refs[s]; refs[s] = (acc + b, n + 1)
                kept += 1
            print(f"[run {run}] {s}: {kept}")

    for s in species:
        write_star(f"{ROOT}/{s}/particles.star", rows[s], box)
        acc, n = refs[s]
        if n:   # fallback average ref (overwritten by build_reference.py with the PDB model)
            with mrcfile.new(f"{ROOT}/{s}/avg_ref.mrc", overwrite=True) as m:
                m.set_data((acc/n).astype(np.float32)); m.voxel_size = APIX
        print(f"== {s}: {len(rows[s])} particles -> {ROOT}/{s}/particles.star")

def g_zarr(run):
    g = glob.glob(f"{DATA}/{run}/*.zarr"); return g[0] if g else run

def write_star(path, rows, box):
    with open(path, "w") as f:
        f.write("\n# version 30001\n\ndata_optics\n\nloop_\n")
        for k, c in enumerate(["_rlnOpticsGroupName","_rlnOpticsGroup","_rlnSphericalAberration",
            "_rlnVoltage","_rlnImagePixelSize","_rlnImageSize","_rlnImageDimensionality",
            "_rlnAmplitudeContrast"], 1): f.write(f"{c} #{k}\n")
        f.write(f"opticsGroup1 1 2.700000 300.000000 {APIX:.6f} {box} 3 0.100000\n")
        f.write("\n# version 30001\n\ndata_particles\n\nloop_\n")
        for k, c in enumerate(["_rlnImageName","_rlnCtfImage","_rlnMicrographName",
            "_rlnCoordinateX","_rlnCoordinateY","_rlnCoordinateZ","_rlnAngleRot","_rlnAngleTilt",
            "_rlnAnglePsi","_rlnOriginXAngst","_rlnOriginYAngst","_rlnOriginZAngst",
            "_rlnOpticsGroup","_rlnGroupNumber","_rlnRandomSubset"], 1): f.write(f"{c} #{k}\n")
        for (img,ctf,mic,cx,cy,cz,grp,subset) in rows:
            f.write(f"{img} {ctf} {mic} {cx:8d} {cy:8d} {cz:8d} "
                    f"{0.0:10.4f} {0.0:10.4f} {0.0:10.4f} 0.0000 0.0000 0.0000 "
                    f"1 {grp:5d} {subset:3d}\n")

if __name__ == "__main__":
    main()

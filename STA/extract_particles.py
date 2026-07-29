#!/usr/bin/env python
"""
Extract subtomograms + write RELION-4 particle STAR files for STA of the
simulated globular proteins in the *basic noise* tomograms (train_dir_6).

Orientations use the GROUND TRUTH from the motif-list CSV via a fully validated
GT->RELION recipe (see README.md). NOTE: the overlay *.json orientations are
BUGGY (scalar-first quaternion read as scalar-last) -- we read the motif-list
CSV directly and rebuild the matrix correctly.

Recipe per particle (verified: reconstruction preserves the coordinate frame,
NO z-flip -- see notes; the earlier z-flip was a VLP hollow-shell artifact):
  R_pn = quaternion matrix from (w,x,y,z)=(Q1,Q2,Q3,Q4)       # scalar-first
  rot,tilt,psi = relion_euler.matrix2angles(R_pn)             # no mirror
  coords: ix=x/apix, iy=y/apix, iz=z/apix                     # no flip

Run:
  micromamba activate pro-revelio
  python extract_particles.py --species all --tomos all
  python extract_particles.py --species ribosome vlp --tomos 0 1     # subset/test
  python extract_particles.py --species vlp --angles zero            # de-novo (no GT)
"""
import argparse, glob, os, sys
import numpy as np, mrcfile, pandas as pd
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relion_euler import matrix2angles

DS    = "/mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_dataset_20260423"
TRAIN = f"{DS}/train_dir_6"
MOTIF = f"{DS}/simulation_dir_6/motif_lists"
ROOT  = f"{DS}/relion_sta"
SIM, APIX = 6, 10.0

# species short-name -> motif-list Code
CODES = {
    "beta-amylase": "pdb_1fa2", "beta-galactosidase": "pdb_6drv",
    "vlp": "pdb_6n4v", "ribosome": "pdb_6qzp", "thyroglobulin": "pdb_7n4y",
    "apo-ferritin": "pdb_8cpv", "albumin": "pdb_8vaf",
}
DIAM = {  # approx particle diameter (A) for RELION --particle_diameter
    "beta-amylase": 130, "beta-galactosidase": 180, "vlp": 320, "ribosome": 300,
    "thyroglobulin": 250, "apo-ferritin": 130, "albumin": 100,
}

def tomo(i, subdir):
    g = glob.glob(f"{TRAIN}/{subdir}/tomogram_{SIM}_{i}/*.mrc"); return g[0] if g else None

def gt_angles(row):
    Rpn = Rotation.from_quat([row.Q2, row.Q3, row.Q4, row.Q1]).as_matrix()   # (w,x,y,z)->xyzw
    return matrix2angles(Rpn)                                                 # no z-flip -> no mirror

def write_wedge(path, box):
    c = box // 2; k, i, j = np.mgrid[0:box, 0:box, 0:box]
    mask = (np.degrees(np.arctan2(np.abs(k-c), np.abs(j-c))) <= 60.0).astype(np.float32)
    with mrcfile.new(path, overwrite=True) as m: m.set_data(mask); m.voxel_size = APIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="+", default=["all"])
    ap.add_argument("--tomos", nargs="+", default=["all"])
    ap.add_argument("--box", type=int, default=64)
    ap.add_argument("--angles", choices=["gt", "zero"], default="gt")
    ap.add_argument("--variant", choices=["basic", "faket", "clean"], default="basic",
                    help="basic = plain-noise tomos; faket = style-transfer tomos; "
                         "clean = noiseless PolNet density (tomo_den, ground-truth reference)")
    ap.add_argument("--no-invert", action="store_true")
    args = ap.parse_args()

    if args.variant == "clean":
        root = f"{DS}/relion_sta_clean"
        cleanp = lambda i: f"{DS}/simulation_dir_6/tomos/tomo_den_{i}.mrc"
        tomo_path = lambda i: cleanp(i) if os.path.exists(cleanp(i)) else None
        list_glob = f"{DS}/simulation_dir_6/tomos/tomo_den_*.mrc"
        idx_of = lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0])
    else:
        subdir = "faket_tomograms" if args.variant == "faket" else "basic_tomograms"
        root = f"{DS}/relion_sta_faket" if args.variant == "faket" else f"{DS}/relion_sta"
        tomo_path = lambda i: tomo(i, subdir)
        list_glob = f"{TRAIN}/{subdir}/tomogram_{SIM}_*"
        idx_of = lambda p: int(os.path.basename(p).split("_")[-1])

    species = list(CODES) if args.species == ["all"] else args.species
    for s in species:
        assert s in CODES, f"unknown species {s}; choose from {list(CODES)}"
    if args.tomos == ["all"]:
        idxs = sorted(idx_of(p) for p in glob.glob(list_glob))
    else:
        idxs = [int(t) for t in args.tomos]

    box, H = args.box, args.box // 2
    invert = (not args.no_invert) and args.variant != "clean"  # clean density already protein-bright
    rows = {s: [] for s in species}       # per species: list of (img, ctf, mic, cx,cy,cz, rot,tilt,psi, grp, subset)
    refs = {s: (np.zeros((box,box,box), np.float64), 0) for s in species}
    for s in species:
        os.makedirs(f"{root}/{s}/Subtomograms", exist_ok=True)
        write_wedge(f"{root}/{s}/wedge_ctf.mrc", box)

    for gi, i in enumerate(idxs):
        tp = tomo_path(i)
        mf = f"{MOTIF}/tomo_motif_list_{i}.csv"
        if not tp or not os.path.exists(mf):
            print(f"[tomo {i}] missing, skip"); continue
        with mrcfile.open(tp, permissive=True) as m: vol = m.data.astype(np.float32).copy()
        nz, ny, nx = vol.shape
        df = pd.read_csv(mf, sep="\t")
        for s in species:
            sub_df = df[df["Code"] == CODES[s]]
            os.makedirs(f"{root}/{s}/Subtomograms/tomogram_{SIM}_{i}", exist_ok=True)
            kept = 0
            for j, row in enumerate(sub_df.itertuples(index=False)):
                ix, iy, iz = int(round(row.X/APIX)), int(round(row.Y/APIX)), int(round(row.Z/APIX))
                if ix-H<0 or ix+H>nx or iy-H<0 or iy+H>ny or iz-H<0 or iz+H>nz: continue
                b = vol[iz-H:iz+H, iy-H:iy+H, ix-H:ix+H].astype(np.float32)
                b = (b - b.mean())/(b.std()+1e-9)
                if invert: b = -b
                rel = f"Subtomograms/tomogram_{SIM}_{i}/{s}_{i:03d}_{j:04d}.mrc"
                with mrcfile.new(f"{root}/{s}/{rel}", overwrite=True) as om:
                    om.set_data(b); om.voxel_size = APIX
                rot, tilt, psi = gt_angles(row) if args.angles == "gt" else (0., 0., 0.)
                rows[s].append((rel, "wedge_ctf.mrc", tp, ix, iy, iz, rot, tilt, psi, gi+1, len(rows[s])%2+1))
                acc, n = refs[s]; refs[s] = (acc + b, n + 1)
                kept += 1
            print(f"[tomo {i}] {s}: {kept}")

    for s in species:
        write_star(f"{root}/{s}/particles.star", rows[s], box)
        acc, n = refs[s]
        if n:
            with mrcfile.new(f"{root}/{s}/initial_ref.mrc", overwrite=True) as m:
                m.set_data((acc/n).astype(np.float32)); m.voxel_size = APIX
        print(f"== {s}: {len(rows[s])} particles -> {root}/{s}/particles.star")

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
        for (img,ctf,mic,cx,cy,cz,rot,tilt,psi,grp,subset) in rows:
            f.write(f"{img} {ctf} {mic} {cx:8d} {cy:8d} {cz:8d} "
                    f"{rot:10.4f} {tilt:10.4f} {psi:10.4f} 0.0000 0.0000 0.0000 "
                    f"1 {grp:5d} {subset:3d}\n")

def write_run_script(s, box):
    diam = DIAM.get(s, box*APIX*0.7)
    with open(f"{ROOT}/{s}/run_refine3d.sh", "w") as f:
        f.write(f"""#!/bin/bash
# RELION 4.0 3D auto-refine of {s}. Run from this dir. (wrap in sbatch for real runs)
module load gcc/13.2.0 openmpi/5.0.7 relion/4.0.1 2>/dev/null || true
cd "$(dirname "$0")"; mkdir -p Refine3D
relion_refine --i particles.star --o Refine3D/run \\
  --ref initial_ref.mrc --ini_high 50 \\
  --auto_refine --split_random_halves --sym C1 \\
  --particle_diameter {int(diam)} --flatten_solvent --zero_mask \\
  --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 \\
  --offset_range 8 --offset_step 2 --pad 2 --tau2_fudge 4 \\
  --dont_combine_weights_via_disc --pool 8 --j 8
# GT angles are already in particles.star. To validate them without re-searching,
# add: --skip_align   (reconstruct straight from the GT orientations).
""")
    os.chmod(f"{ROOT}/{s}/run_refine3d.sh", 0o750)

if __name__ == "__main__":
    main()

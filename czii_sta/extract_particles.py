#!/usr/bin/env python
"""
Cut subtomograms out of the CZII OME-Zarr tomograms and write a RELION-4 particle
STAR file, plus the shared missing-wedge volume used as rlnCtfImage.

All conventions below were measured from the data, not assumed:
  pixel size 10.012 A      (zarr .zattrs)
  protein is DARK          (density at picks is 2-4 sigma below the global mean)
  picks carry NO angles    (every transformation_ is the identity) -> de novo refine
  wedge +-45 deg about Y   (portal tilt_range 90; confirmed by where the power
                            spectrum falls off in the kX-kZ vs kY-kZ planes)

  python extract_particles.py --fraction 0.15 --box 64
"""
import argparse, glob, json, os
import numpy as np, mrcfile, zarr

DATA = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge"
APIX = 10.012
TILT_HALF_RANGE = 45.0          # portal: tilt_range 90 deg, uniform across runs


def wedge(box, half_range):
    """Binary missing-wedge volume, centred, tilt axis along Y."""
    c = box // 2
    k, _, j = np.mgrid[0:box, 0:box, 0:box]
    ang = np.degrees(np.arctan2(np.abs(k - c), np.abs(j - c)))
    return (ang <= half_range).astype(np.float32)


def collect(species, runs):
    """[(run, x, y, z), ...] in Angstrom, for every pick of this species."""
    out = []
    for r in runs:
        f = f"{DATA}/ground_truth/{r}/Picks/{species}.json"
        if not os.path.isfile(f):
            continue
        for p in json.load(open(f)).get("points", []):
            L = p["location"]
            out.append((r, L["x"], L["y"], L["z"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="ribosome")
    ap.add_argument("--box", type=int, default=64)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="random fraction of particles to keep (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_sta_ribosome")
    args = ap.parse_args()

    # ground_truth/ also holds unrelated simulation dirs (tomogram_0_*,
    # basic_noise_simulation, ...). Only runs that have BOTH picks and a tomogram
    # are usable, and the subsample must be drawn from those alone.
    gt = set(os.listdir(f"{DATA}/ground_truth"))
    tm = set(os.listdir(f"{DATA}/data/tomograms_VoxelSpacing10"))
    runs = sorted(gt & tm)
    print(f"{len(gt)} ground_truth dirs, {len(tm)} tomograms -> {len(runs)} usable runs "
          f"({len(gt - tm)} non-CZII dirs ignored)")
    picks = collect(args.species, runs)
    print(f"{len(picks)} {args.species} picks across {len(runs)} runs")

    rng = np.random.default_rng(args.seed)
    if args.fraction < 1.0:
        # drawn across all runs at once, so no single tomogram is over-represented
        keep = rng.choice(len(picks), int(round(len(picks) * args.fraction)), replace=False)
        picks = [picks[i] for i in sorted(keep)]
        print(f"  -> keeping {len(picks)} ({args.fraction:.0%}, seed {args.seed})")

    os.makedirs(f"{args.out}/Subtomograms", exist_ok=True)
    with mrcfile.new(f"{args.out}/wedge_ctf.mrc", overwrite=True) as m:
        m.set_data(wedge(args.box, TILT_HALF_RANGE)); m.voxel_size = APIX

    H = args.box // 2
    by_run = {}
    for r, x, y, z in picks:
        by_run.setdefault(r, []).append((x, y, z))

    rows, edge = [], 0
    for gi, (r, pts) in enumerate(sorted(by_run.items()), start=1):
        d = f"{DATA}/data/tomograms_VoxelSpacing10/{r}"
        zs = [p for p in os.listdir(d) if p.endswith(".zarr")] if os.path.isdir(d) else []
        if not zs:
            print(f"  [run {r}] no zarr, skipped"); continue
        vol = np.asarray(zarr.open(f"{d}/{zs[0]}", mode="r")["0"]).astype(np.float32)
        nz, ny, nx = vol.shape
        os.makedirs(f"{args.out}/Subtomograms/{r}", exist_ok=True)
        kept = 0
        for n, (x, y, z) in enumerate(pts):
            ix, iy, iz = (int(round(v / APIX)) for v in (x, y, z))
            if ix-H < 0 or ix+H > nx or iy-H < 0 or iy+H > ny or iz-H < 0 or iz+H > nz:
                edge += 1; continue
            b = vol[iz-H:iz+H, iy-H:iy+H, ix-H:ix+H]
            b = -(b - b.mean()) / (b.std() + 1e-9)      # invert: protein is dark
            rel = f"Subtomograms/{r}/{args.species}_{r}_{n:05d}.mrc"
            with mrcfile.new(f"{args.out}/{rel}", overwrite=True) as m:
                m.set_data(b.astype(np.float32)); m.voxel_size = APIX
            rows.append((rel, zs[0], ix, iy, iz, gi))
            kept += 1
        del vol                      # 121 volumes at ~300 MB each; don't hold them
        print(f"  [run {r}] {kept}")

    subset = rng.integers(1, 3, size=len(rows))          # reproducible half-set split
    write_star(f"{args.out}/particles.star", rows, subset, args.box)
    print(f"\n{len(rows)} particles written ({edge} dropped at box edges)")
    print(f"  half-sets: {(subset==1).sum()} / {(subset==2).sum()}")
    print(f"  -> {args.out}/particles.star")


def write_star(path, rows, subset, box):
    with open(path, "w") as f:
        f.write("\n# version 30001\n\ndata_optics\n\nloop_\n")
        for k, c in enumerate(["_rlnOpticsGroupName", "_rlnOpticsGroup", "_rlnSphericalAberration",
                               "_rlnVoltage", "_rlnImagePixelSize", "_rlnImageSize",
                               "_rlnImageDimensionality", "_rlnAmplitudeContrast"], 1):
            f.write(f"{c} #{k}\n")
        f.write(f"opticsGroup1 1 2.700000 300.000000 {APIX:.6f} {box} 3 0.100000\n")
        f.write("\n# version 30001\n\ndata_particles\n\nloop_\n")
        for k, c in enumerate(["_rlnImageName", "_rlnCtfImage", "_rlnMicrographName",
                               "_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ",
                               "_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi",
                               "_rlnOriginXAngst", "_rlnOriginYAngst", "_rlnOriginZAngst",
                               "_rlnOpticsGroup", "_rlnGroupNumber", "_rlnRandomSubset"], 1):
            f.write(f"{c} #{k}\n")
        for (rel, mic, ix, iy, iz, gi), s in zip(rows, subset):
            f.write(f"{rel} wedge_ctf.mrc {mic} {ix:7d} {iy:7d} {iz:7d} "
                    f"0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1 {gi:5d} {s:3d}\n")


if __name__ == "__main__":
    main()

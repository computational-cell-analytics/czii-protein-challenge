#!/usr/bin/env python
"""
Per-variant RESOLUTION / QUALITY, two complementary measures:

  (1) half-map FSC  (PRECISION): split each variant's particles into two random
      halves, reconstruct each half independently at the GT angles, FSC the halves.
      Resolution = FSC 0.143.  (gold-standard cryo-EM resolution)
  (2) map-to-truth FSC (ACCURACY): FSC of each variant's full GT-angle average
      against the noiseless clean-density reconstruction (PolNet tomo_den, same
      GT angles / same box / same wedge -> same frame). Resolution = FSC 0.5.

Prereq: the clean reference must be extracted once:
    python extract_particles.py --species all --tomos all --variant clean
then:
    python resolution_report.py

Outputs (STA/results/resolution/):
    resolution_summary.csv   variant, species, N, halfmap_0.143, accuracy_0.5, accuracy_0.143
    fsc_quality_<species>.svg   half-map + vs-truth curves, basic & faket
    report.txt
"""
import argparse, os, glob, subprocess, sys
import numpy as np, mrcfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_fsc import (DATASET, SPECIES_ALL, DIAM, VARIANT_DIR,
                         load, soft_sphere, fsc_curve, crossing_res, svg_plot)

# Use whatever relion_refine is on PATH (load the cluster module first, e.g.
# `source env.sh` / `module load gcc/13.2.0 openmpi/5.0.7 relion/4.0.1`).
RELION = "relion_refine"
RESULTS = "/user/muth9/u12095/czii-protein-challenge/STA/results/resolution"
GTMAP = "GTaverage/run_it001_class001.mrc"

def reconstruct(projdir, star, outprefix, diam):
    os.makedirs(f"{projdir}/{os.path.dirname(outprefix)}", exist_ok=True)
    subprocess.run([RELION, "--i", star, "--o", outprefix, "--ref", "initial_ref.mrc",
        "--ini_high", "30", "--skip_align", "--iter", "1", "--K", "1", "--sym", "C1",
        "--particle_diameter", str(diam), "--flatten_solvent", "--zero_mask",
        "--pad", "2", "--tau2_fudge", "4", "--dont_combine_weights_via_disc", "--j", "8"],
        cwd=projdir, capture_output=True, text=True)
    m = f"{projdir}/{outprefix}_it001_class001.mrc"
    return m if os.path.exists(m) else None

def split_star(star, h1, h2):
    lines = open(star).read().splitlines()
    ds = next(i for i, l in enumerate(lines) if "Subtomograms/" in l)
    pre, rows = lines[:ds], [l for l in lines[ds:] if "Subtomograms/" in l]
    open(h1, "w").write("\n".join(pre + [r for r in rows if r.split()[-1] == "1"]) + "\n")
    open(h2, "w").write("\n".join(pre + [r for r in rows if r.split()[-1] == "2"]) + "\n")

def cleanup(projdir, *names):
    for n in names:
        subprocess.run(["rm", "-rf", f"{projdir}/{n}"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="+", default=SPECIES_ALL)
    ap.add_argument("--variants", nargs="+", default=["basic", "faket"])
    ap.add_argument("--box", type=int, default=64)
    ap.add_argument("--apix", type=float, default=10.0)
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()
    box, apix = args.box, args.apix
    os.makedirs(args.out, exist_ok=True)
    COL = {"basic": "#0072B2", "faket": "#E69F00"}
    COLT = {"basic": "#009E73", "faket": "#D55E00"}

    rows = []                                   # (variant, species, N, half143, acc05, acc143)
    curves = {sp: [] for sp in args.species}
    for sp in args.species:
        diam = DIAM.get(sp, 250)
        mask = soft_sphere(box, DIAM.get(sp, box*apix*0.35)/2/apix + 3)
        # ---- clean (ground-truth) reference, reconstructed like the variants ----
        cdir = f"{DATASET}/relion_sta_clean/{sp}"
        clean_ref = None
        if os.path.exists(f"{cdir}/particles.star"):
            clean_ref = reconstruct(cdir, "particles.star", "_cleanref/run", diam)
        if clean_ref is None:
            print(f"[warn] {sp}: no clean reference (run extract --variant clean); accuracy skipped")

        for variant in args.variants:
            projdir = f"{DATASET}/{VARIANT_DIR[variant]}/{sp}"
            star = f"{projdir}/particles.star"
            if not os.path.exists(star):
                print(f"[skip] {variant}/{sp}: no particles.star"); continue
            # (1) half-map FSC
            split_star(star, f"{projdir}/_h1.star", f"{projdir}/_h2.star")
            m1 = reconstruct(projdir, "_h1.star", "_H1/run", diam)
            m2 = reconstruct(projdir, "_h2.star", "_H2/run", diam)
            half143 = np.nan; half_fsc = None
            if m1 and m2:
                half_fsc = fsc_curve(load(m1), load(m2), mask, box)
                half143 = crossing_res(half_fsc, box, apix, 0.143)
            n = sum(1 for l in open(star) if "Subtomograms/" in l)
            # (2) accuracy FSC vs clean truth (use the full GT-angle average already made)
            acc05 = acc143 = np.nan; acc_fsc = None
            gt = f"{projdir}/{GTMAP}"
            if clean_ref and os.path.exists(gt):
                acc_fsc = fsc_curve(load(gt), load(clean_ref), mask, box)
                acc05, acc143 = crossing_res(acc_fsc, box, apix, 0.5), crossing_res(acc_fsc, box, apix, 0.143)
            rows.append((variant, sp, n, half143, acc05, acc143))
            freqs = np.arange(1, (box//2))/(box*apix)
            if half_fsc is not None:
                curves[sp].append((f"{variant} (half-map)", COL[variant], freqs, half_fsc))
            if acc_fsc is not None:
                curves[sp].append((f"{variant} (vs truth)", COLT[variant], freqs, acc_fsc))
            print(f"{variant:6s} {sp:20s} N={n:5d}  half-map(0.143)={half143:5.0f}A  vs-truth(0.5)={acc05:5.0f}A")
            cleanup(projdir, "_h1.star", "_h2.star", "_H1", "_H2")
        if clean_ref: cleanup(cdir, "_cleanref")

    for sp in args.species:
        if curves[sp]:
            svg_plot(curves[sp], box, apix, f"{args.out}/fsc_quality_{sp}.svg",
                     f"Resolution/quality FSC: {sp}")
    with open(f"{args.out}/resolution_summary.csv", "w") as f:
        f.write("variant,species,n_particles,halfmap_res_0.143_A,accuracy_res_0.5_A,accuracy_res_0.143_A\n")
        for v, sp, n, h143, a05, a143 in rows:
            f.write(f"{v},{sp},{n},{h143:.1f},{a05:.1f},{a143:.1f}\n")
    with open(f"{args.out}/report.txt", "w") as f:
        f.write("Per-variant RESOLUTION / QUALITY\n")
        f.write(f"box={box} apix={apix}A -> Nyquist={2*apix:.0f}A\n\n")
        f.write(f"{'variant':7s} {'species':20s} {'N':>6s} {'half-map 0.143':>14s} {'vs-truth 0.5':>13s} {'vs-truth 0.143':>15s}\n")
        for v, sp, n, h143, a05, a143 in rows:
            f.write(f"{v:7s} {sp:20s} {n:6d} {h143:14.0f} {a05:13.0f} {a143:15.0f}\n")
        f.write("""
NOTES
- half-map FSC (0.143) = PRECISION: split each variant's data in two, reconstruct
  each half at GT angles, FSC. Standard gold-standard resolution.
- vs-truth FSC = ACCURACY: variant's full GT-angle average vs the noiseless
  clean-density (PolNet tomo_den) reconstruction, same frame/wedge. Read at 0.5
  (map-to-model convention); 0.143 also given.
- Both reconstructed at GT angles (--skip_align) -> quality at known-correct poses.
  A de-novo auto-refine (run_all.sh refine) would measure resolution when RELION
  must FIND the poses. 10 A/voxel -> Nyquist 20 A.
- A value of "20" (=Nyquist) means the FSC never crossed the threshold within the
  measurable range (resolution AT the sampling limit, not precise). High half-map
  FSC = PRECISION (reproducible), which is NOT the same as ACCURACY (vs-truth):
  clean/style-transfer data can be very reproducible yet only ~30 A accurate.
""")
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()

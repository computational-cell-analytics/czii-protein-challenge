#!/usr/bin/env python
"""
Experimental-data FSC comparisons that were missing:

  (A) EXPERIMENTAL half-map precision, in-house NUMPY method (the RELION version
      is the auto-refine gold-standard FSC in run_model.star, done by
      relion_fsc_report.py). FSC of the two auto-refine half-maps
      (run_half1/2_class001_unfil.mrc) -- they share the refine's frame, so NO
      alignment is needed. Read at 0.143 (gold-standard).

  (B) SIM-vs-EXPERIMENTAL cross-FSC, for BOTH basic and faket, analogous to the
      basic-vs-faket cross-FSC -- BUT the sim GT-average (ground-truth poses) and
      the experimental de-novo map live in DIFFERENT frames, so they must be
      rigid-body aligned first (rotation search + mirror check + COM centring).
      Computed BOTH ways: numpy fsc_curve AND relion_postprocess. Read at 0.5
      (comparing two different maps).

  ! Caveat: at 10 A the densities are near-spherical (low orientational contrast),
    so the rigid alignment in (B) is approximate; treat sim-vs-exp cross-FSC as an
    overall-shape-similarity indicator, not a precise resolution.

    micromamba activate pro-revelio
    module load gcc/13.2.0 openmpi/5.0.7 relion/4.0.1     # for the RELION part
    python exp_compare_fsc.py                              # all species
    python exp_compare_fsc.py --species ribosome --no-relion

Outputs (STA/results/exp_compare/):
    summary.csv    method, comparison, species, N, cc_aligned, mirror, res_0.5_A, res_0.143_A
    curves.csv     full FSC curves (long)
    report.txt
    pp/            relion_postprocess runs for the cross-FSCs
"""
import argparse, os, sys
import numpy as np, mrcfile
from scipy.ndimage import affine_transform, center_of_mass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_fsc import DATASET, DIAM, VARIANT_DIR, soft_sphere, fsc_curve, crossing_res, load
from relion_fsc_report import postprocess, parse_pp_star, crossing, pick_fsc, refine_dir

OUT = "/user/muth9/u12095/czii-protein-challenge/STA/results/exp_compare"
EXP_DIR = f"{DATASET}/relion_sta_experimental"
EXP_NAME = {"vlp": "virus-like-particle"}          # sim name -> experimental dir name
SIM_SPECIES = ["ribosome", "vlp", "beta-galactosidase", "thyroglobulin",
               "apo-ferritin", "beta-amylase"]     # albumin has no experimental data
GTMAP = "GTaverage/run_it001_class001.mrc"


# ------------------------------------------------------- rigid map alignment
def rotate_vol(v, R, center):
    """Apply rotation matrix R (world) about `center` to volume v."""
    offset = center - R @ center
    return affine_transform(v, R, offset=offset, order=1, mode="constant", cval=0.0)


def euler_matrix(a, b, c):
    """ZYZ intrinsic euler -> rotation matrix (radians)."""
    ca, sa = np.cos(a), np.sin(a); cb, sb = np.cos(b), np.sin(b); cc, sc = np.cos(c), np.sin(c)
    Rz1 = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz2 = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rz1 @ Ry @ Rz2


def com_center(v, mask):
    """Shift v so the masked centre-of-mass sits at the box centre."""
    box = v.shape[0]; c = (box - 1) / 2.0
    m = np.clip(v, 0, None) * mask
    if m.sum() <= 0:
        return v
    com = np.array(center_of_mass(m))
    return affine_transform(v, np.eye(3), offset=(com - c), order=1, mode="constant", cval=0.0)


def masked_cc(a, b, mask):
    x = (a * mask).ravel(); y = (b * mask).ravel()
    x = x - x.mean(); y = y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 0 else -1.0


def best_align(ref, mov, mask, try_mirror=True):
    """Rigid-align mov onto ref by a coarse->fine ZYZ rotation search (+ optional
    mirror). Both are COM-centred first. Returns (aligned_mov, cc, mirrored)."""
    box = ref.shape[0]; center = np.array([(box-1)/2.0]*3)
    ref_c = com_center(ref, mask)
    cands = [(False, com_center(mov, mask))]
    if try_mirror:
        cands.append((True, com_center(mov[::-1].copy(), mask)))

    best = (-2.0, None, False)
    for mirrored, mv in cands:
        # coarse grid (45 deg), then refine twice around the best
        grid = [np.deg2rad(np.arange(0, 360, 45)),
                np.deg2rad(np.arange(0, 181, 45)),
                np.deg2rad(np.arange(0, 360, 45))]
        cur = None
        for stage, step in enumerate([45, 15, 5]):
            if stage > 0:
                s = np.deg2rad(step)
                grid = [np.array([cur[0]-2*s, cur[0]-s, cur[0], cur[0]+s, cur[0]+2*s]),
                        np.array([cur[1]-2*s, cur[1]-s, cur[1], cur[1]+s, cur[1]+2*s]),
                        np.array([cur[2]-2*s, cur[2]-s, cur[2], cur[2]+s, cur[2]+2*s])]
            local = (-2.0, None)
            for a in grid[0]:
                for b in grid[1]:
                    for c in grid[2]:
                        R = euler_matrix(a, b, c)
                        cc = masked_cc(ref_c, rotate_vol(mv, R, center), mask)
                        if cc > local[0]:
                            local = (cc, (a, b, c))
            cur = local[1]
            if local[0] > best[0]:
                best = (local[0], rotate_vol(mv, euler_matrix(*cur), center), mirrored)
    return best[1], best[0], best[2]


# ------------------------------------------------------- fsc helpers
def save_mrc(path, v, apix):
    with mrcfile.new(path, overwrite=True) as f:
        f.set_data(v.astype(np.float32)); f.voxel_size = apix


def n_particles(star):
    try:
        return sum(1 for l in open(star) if "Subtomograms/" in l)
    except OSError:
        return None


# ------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="+", default=SIM_SPECIES)
    ap.add_argument("--variants", nargs="+", default=["basic", "faket"])
    ap.add_argument("--box", type=int, default=64)
    ap.add_argument("--apix", type=float, default=10.0)
    ap.add_argument("--apix-exp", type=float, default=10.012)
    ap.add_argument("--no-relion", action="store_true", help="skip relion_postprocess parts")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    box, apix = args.box, args.apix
    os.makedirs(f"{args.out}/pp", exist_ok=True)
    os.makedirs(f"{args.out}/aligned", exist_ok=True)
    rows, curves = [], []

    for sp in args.species:
        ename = EXP_NAME.get(sp, sp)
        d = DIAM.get(sp, 250)
        rvox = d/2/apix + 3
        mask = soft_sphere(box, rvox)
        maskpath = f"{args.out}/aligned/mask_{sp}.mrc"; save_mrc(maskpath, mask, apix)
        edir = f"{EXP_DIR}/{ename}/{refine_dir(ename)}"   # symmetry run for VLP(I)/apo(O)

        # ---- (A) experimental half-map precision, NUMPY ----
        h1, h2 = f"{edir}/run_half1_class001_unfil.mrc", f"{edir}/run_half2_class001_unfil.mrc"
        if os.path.exists(h1) and os.path.exists(h2):
            fsc = fsc_curve(load(h1), load(h2), mask, box)
            r143 = crossing_res(fsc, box, apix, 0.143); r05 = crossing_res(fsc, box, apix, 0.5)
            rows.append(("numpy", "exp-halfmap", ename, None, np.nan, "-", r05, r143))
            freqs = np.arange(1, len(fsc)+1)/(box*apix)
            curves.append(("numpy", "exp-halfmap", ename, 1.0/freqs, fsc))
            print(f"[A] numpy exp-halfmap {ename:22s} 0.143={r143:.1f} 0.5={r05:.1f}")

        # ---- (B) sim-vs-experimental cross-FSC (aligned), numpy + RELION ----
        expmap = f"{edir}/run_class001.mrc"
        if not os.path.exists(expmap):
            continue
        exp = load(expmap)
        for variant in args.variants:
            gt = f"{DATASET}/{VARIANT_DIR[variant]}/{sp}/{GTMAP}"
            star = f"{DATASET}/{VARIANT_DIR[variant]}/{sp}/particles.star"
            if not os.path.exists(gt):
                continue
            ref = load(gt)                      # sim GT-average is the reference frame
            aligned, cc, mirrored = best_align(ref, exp, mask)
            N = n_particles(star)
            # numpy FSC of aligned pair
            fsc = fsc_curve(ref, aligned, mask, box)
            r05 = crossing_res(fsc, box, apix, 0.5); r143 = crossing_res(fsc, box, apix, 0.143)
            rows.append(("numpy", f"{variant}-vs-exp", ename, N, cc, "Y" if mirrored else "N", r05, r143))
            freqs = np.arange(1, len(fsc)+1)/(box*apix)
            curves.append(("numpy", f"{variant}-vs-exp", ename, 1.0/freqs, fsc))
            print(f"[B] numpy {variant}-vs-exp {ename:20s} cc={cc:.3f} mirror={'Y' if mirrored else 'N'} 0.5={r05:.1f}")

            # RELION postprocess of the SAME aligned pair
            if not args.no_relion:
                rp = f"{args.out}/aligned/{variant}_vs_exp_{ename}_ref.mrc"
                ap_ = f"{args.out}/aligned/{variant}_vs_exp_{ename}_exp.mrc"
                save_mrc(rp, ref, apix); save_mrc(ap_, aligned, apix)
                star_pp, r = postprocess(rp, ap_, maskpath, apix,
                                         f"{args.out}/pp/{variant}_vs_exp_{ename}/pp")
                if star_pp:
                    cols, final = parse_pp_star(star_pp)
                    fscr = pick_fsc(cols, ["rlnFourierShellCorrelationMaskedMaps",
                                           "rlnFourierShellCorrelationCorrected"])
                    ar = cols.get("rlnAngstromResolution")
                    if fscr is not None and ar is not None:
                        rr05 = crossing(ar, fscr, 0.5); rr143 = crossing(ar, fscr, 0.143)
                        rows.append(("relion", f"{variant}-vs-exp", ename, N, cc,
                                     "Y" if mirrored else "N", rr05, rr143))
                        curves.append(("relion", f"{variant}-vs-exp", ename, ar, fscr))
                        print(f"[B] relion {variant}-vs-exp {ename:20s} 0.5={rr05:.1f}")
                else:
                    print(f"[warn] relion cross failed {variant}/{ename}")

    # ---- write outputs ----
    with open(f"{args.out}/summary.csv", "w") as f:
        f.write("method,comparison,species,n_particles,cc_aligned,mirror,res_0.5_A,res_0.143_A\n")
        for meth, comp, sp, n, cc, mir, r05, r143 in rows:
            ccs = "" if (cc is None or (isinstance(cc, float) and np.isnan(cc))) else f"{cc:.4f}"
            f.write(f"{meth},{comp},{sp},{n if n is not None else ''},{ccs},{mir},{r05:.1f},{r143:.1f}\n")
    with open(f"{args.out}/curves.csv", "w") as f:
        f.write("method,comparison,species,resolution_A,fsc\n")
        for meth, comp, sp, ar, fsc in curves:
            for a, v in zip(ar, fsc):
                f.write(f"{meth},{comp},{sp},{a:.3f},{v:.4f}\n")
    with open(f"{args.out}/report.txt", "w") as f:
        f.write("Experimental-data FSC: numpy + RELION\n")
        f.write(f"box={box} apix={apix}A (exp {args.apix_exp}A) -> Nyquist={2*apix:.0f}A\n\n")
        f.write(f"{'method':7s} {'comparison':16s} {'species':22s} {'N':>6s} {'cc':>7s} {'mir':>3s} {'0.5(A)':>8s} {'0.143(A)':>9s}\n")
        for meth, comp, sp, n, cc, mir, r05, r143 in rows:
            ccs = "  -  " if (cc is None or (isinstance(cc, float) and np.isnan(cc))) else f"{cc:6.3f}"
            f.write(f"{meth:7s} {comp:16s} {sp:22s} {str(n or '-'):>6s} {ccs:>7s} {mir:>3s} {r05:8.1f} {r143:9.1f}\n")
        f.write("""
NOTES
- exp-halfmap (numpy): PRECISION of the experimental de-novo refine, computed by
  the in-house fsc_curve on the two auto-refine half-maps. Read at 0.143. The
  RELION counterpart is the gold-standard FSC in run_model.star (relion_fsc_report).
  There is NO vs-truth/accuracy for experimental data -- real data has no
  ground-truth density.
- {basic,faket}-vs-exp: cross-FSC of the sim GT-average vs the experimental map.
  The two are in DIFFERENT frames (GT poses vs de-novo), so the experimental map
  was rigid-body aligned to the sim average first (rotation search + mirror + COM
  centre). 'cc' = masked real-space correlation AFTER alignment; 'mir'=Y means a
  mirror gave the better fit. Read FSC at 0.5 (two different maps).
- CAVEAT: near-spherical densities at 10 A => low orientational contrast => the
  alignment is approximate. Treat {basic,faket}-vs-exp as overall-shape similarity,
  not a precise resolution. Low 'cc' (e.g. VLP, whose de-novo refine did not
  converge) means the comparison is unreliable for that species.
- numpy vs relion rows use the SAME aligned maps; differences are due to RELION's
  solvent-mask/phase-randomization corrections vs the numpy soft-sphere mask.
""")
    print(f"\nWrote experimental FSC comparison -> {args.out}")


if __name__ == "__main__":
    main()

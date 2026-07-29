#!/usr/bin/env python
"""
RELION-native FSC report -- the SAME comparisons as compare_fsc.py /
resolution_report.py, but the FSC is computed by `relion_postprocess`
(RELION's own masked + phase-randomization-corrected gold-standard FSC)
instead of the in-house numpy fsc_curve. Saved to a SEPARATE folder so the
numpy results are left untouched, for a side-by-side cross-check.

Comparisons (all via relion_postprocess --i <A> --i2 <B> --mask <soft sphere>):
  (1) half-map PRECISION   basic & faket: split particles, reconstruct each half
                           at GT angles, postprocess the two halves. Resolution =
                           RELION's FinalResolution (corrected FSC @0.143).
  (2) vs-truth ACCURACY    each variant's GT average  vs  the clean-density
                           (PolNet tomo_den) GT reconstruction. Masked FSC @0.5.
  (3) basic-vs-faket       the two variants' GT averages. Masked FSC @0.5.
  (4) experimental         de-novo auto-refine half-maps (already made by RELION);
                           postprocess them. Corrected FSC @0.143 + run_model res.

Prereq: RELION 4 on PATH (module load gcc/13.2.0 openmpi/5.0.7 relion/4.0.1),
run in an env with numpy+mrcfile (pro-revelio is fine; only numpy/mrcfile used).

    python relion_fsc_report.py                 # everything
    python relion_fsc_report.py --species ribosome vlp
    python relion_fsc_report.py --skip-recon    # reuse half/clean maps if present

Outputs (STA/results/relion_fsc/):
    resolution_summary.csv   variant/comparison, species, N, resolutions
    fsc_curves.csv           full FSC curve (long) for every comparison
    report.txt               human-readable summary + notes
    pp/<...>/                each relion_postprocess run (postprocess.star kept)
    masks/<species>.mrc      the soft-sphere masks used
"""
import argparse, os, subprocess, sys, glob
import numpy as np, mrcfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_fsc import DATASET, DIAM, VARIANT_DIR, soft_sphere

RELION_REFINE = "relion_refine"
RELION_PP = "relion_postprocess"
OUT = "/user/muth9/u12095/czii-protein-challenge/STA/results/relion_fsc"
GTMAP = "GTaverage/run_it001_class001.mrc"
# sim species (basic/faket/clean use "vlp"); experimental uses "virus-like-particle"
SIM_SPECIES = ["ribosome", "vlp", "beta-galactosidase", "thyroglobulin",
               "apo-ferritin", "beta-amylase", "albumin"]
EXP_DIR = f"{DATASET}/relion_sta_experimental"
EXP_NAME = {"vlp": "virus-like-particle"}          # sim name -> experimental dir name
# Which auto-refine folder to use per experimental species. The default C1 refine
# (Refine3D) FAILED for the symmetric particles (VLP 128A, apo-ferritin poor), so
# those use their symmetry-imposed runs instead (run_refine3d_sym.sh):
#   VLP -> icosahedral (Refine3D_I),  apo-ferritin -> octahedral (Refine3D_O).
EXP_REFINE = {"virus-like-particle": "Refine3D_I", "apo-ferritin": "Refine3D_O"}
def refine_dir(ename):
    return EXP_REFINE.get(ename, "Refine3D")


# ---------------------------------------------------------------- helpers
def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r


def write_mask(path, box, rvox, apix):
    m = soft_sphere(box, rvox)
    with mrcfile.new(path, overwrite=True) as f:
        f.set_data(m.astype(np.float32)); f.voxel_size = apix


def reconstruct(projdir, star, outprefix, diam):
    """GT-angle reconstruction (no alignment), like resolution_report.py."""
    os.makedirs(f"{projdir}/{os.path.dirname(outprefix)}", exist_ok=True)
    run([RELION_REFINE, "--i", star, "--o", outprefix, "--ref", "initial_ref.mrc",
         "--ini_high", "30", "--skip_align", "--iter", "1", "--K", "1", "--sym", "C1",
         "--particle_diameter", str(diam), "--flatten_solvent", "--zero_mask",
         "--pad", "2", "--tau2_fudge", "4", "--dont_combine_weights_via_disc", "--j", "8"],
        cwd=projdir)
    m = f"{projdir}/{outprefix}_it001_class001.mrc"
    return m if os.path.exists(m) else None


def split_star(star, h1, h2):
    lines = open(star).read().splitlines()
    ds = next(i for i, l in enumerate(lines) if "Subtomograms/" in l)
    pre, rows = lines[:ds], [l for l in lines[ds:] if "Subtomograms/" in l]
    open(h1, "w").write("\n".join(pre + [r for r in rows if r.split()[-1] == "1"]) + "\n")
    open(h2, "w").write("\n".join(pre + [r for r in rows if r.split()[-1] == "2"]) + "\n")


def n_particles(star):
    try:
        return sum(1 for l in open(star) if "Subtomograms/" in l)
    except OSError:
        return None


def postprocess(map_a, map_b, mask, apix, out_root):
    """Run relion_postprocess; if the FSC never drops below the phase-randomization
    threshold (very correlated maps), retry with progressively lower thresholds."""
    os.makedirs(os.path.dirname(out_root), exist_ok=True)
    star = f"{out_root}.star"
    for rz in ("0.8", "0.5", "0.3", "0.1"):
        r = run([RELION_PP, "--i", map_a, "--i2", map_b, "--mask", mask,
                 "--angpix", str(apix), "--randomize_at_fsc", rz, "--o", out_root])
        if os.path.exists(star):
            return star, r
        if "never drops below" not in (r.stdout + r.stderr):
            break                     # a different error -> stop retrying
    return None, r


def parse_pp_star(star):
    """Return (dict col->np.array for data_fsc, final_resolution or nan)."""
    lines = open(star).read().splitlines()
    cols, data, final = [], [], np.nan
    # data_general: FinalResolution
    for i, l in enumerate(lines):
        if l.strip().startswith("_rlnFinalResolution"):
            toks = l.split()
            if len(toks) >= 2 and not toks[1].startswith("#"):
                final = float(toks[1])
    # data_fsc block
    in_fsc = False; header = []
    for l in lines:
        s = l.strip()
        if s == "data_fsc":
            in_fsc = True; header = []; data = []; continue
        if in_fsc:
            if s.startswith("_rln"):
                header.append(s.split()[0][1:]); continue    # strip leading _
            if s.startswith("loop_") or s == "":
                if data:                                       # blank after data ends block
                    break
                continue
            if s.startswith("data_"):
                break
            parts = s.split()
            if len(parts) == len(header):
                data.append([float(x) for x in parts])
    arr = np.array(data) if data else np.zeros((0, len(header)))
    return {h: arr[:, i] for i, h in enumerate(header)} if arr.size else {}, final


def crossing(ang_res, fsc, thr):
    """First (highest-res) crossing below thr, linear-interp in freq. ang_res
    descending along the array (low freq first)."""
    freq = 1.0 / np.asarray(ang_res)
    fsc = np.asarray(fsc)
    for i in range(len(fsc)):
        if fsc[i] < thr:
            if i == 0:
                return float(ang_res[0])
            f0, f1, y0, y1 = freq[i-1], freq[i], fsc[i-1], fsc[i]
            fc = f0 + (f1-f0)*(y0-thr)/(y0-y1) if y0 != y1 else f1
            return float(1.0/fc)
    return float(ang_res[-1])          # never crosses -> Nyquist-ish (last shell)


def parse_model_star(star):
    """RELION auto-refine run_model.star -> (ang_res[], goldStandardFsc[], currentRes).
    Reads the data_model_class_1 loop (gold-standard FSC = RELION's native refine FSC)."""
    lines = open(star).read().splitlines()
    cur = np.nan
    for l in lines:
        if l.strip().startswith("_rlnCurrentResolution"):
            cur = float(l.split()[1])
    # locate data_model_class_1 loop
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == "data_model_class_1")
    except StopIteration:
        return None, None, cur
    header, rows, reading = [], [], False
    for l in lines[start+1:]:
        s = l.strip()
        if s.startswith("_rln"):
            header.append(s.split()[0][1:]); continue
        if s == "loop_" or s == "":
            if reading:
                break
            continue
        if s.startswith("data_"):
            break
        parts = s.split()
        if len(parts) == len(header):
            try:
                rows.append([float(x) for x in parts]); reading = True
            except ValueError:
                break
        elif reading:
            break
    if not rows:
        return None, None, cur
    arr = np.array(rows)
    col = {h: arr[:, i] for i, h in enumerate(header)}
    return col.get("rlnAngstromResolution"), col.get("rlnGoldStandardFsc"), cur


def pick_fsc(cols, prefer):
    """Choose an FSC column from a postprocess data_fsc dict."""
    for k in prefer:
        if k in cols:
            return cols[k]
    return None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="+", default=SIM_SPECIES)
    ap.add_argument("--variants", nargs="+", default=["basic", "faket"])
    ap.add_argument("--box", type=int, default=64)
    ap.add_argument("--apix", type=float, default=10.0)
    ap.add_argument("--apix-exp", type=float, default=10.012)
    ap.add_argument("--skip-recon", action="store_true",
                    help="reuse existing half/clean recons if present")
    ap.add_argument("--experimental-only", action="store_true",
                    help="skip the sim reconstructions; recompute only the experimental "
                         "rows and MERGE them into the existing CSVs/report")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    box, apix = args.box, args.apix
    os.makedirs(f"{args.out}/masks", exist_ok=True)
    os.makedirs(f"{args.out}/pp", exist_ok=True)

    rows = []          # (comparison, species, N, res_primary, thr_primary, res_0.5, res_0.143)
    curves = []        # (comparison, species, ang_res, fsc)

    # masks per species (box 64), radius from particle diameter
    masks = {}
    for sp in set(args.species) | {EXP_NAME.get(s, s) for s in args.species}:
        d = DIAM.get(sp, DIAM.get("vlp") if sp == "virus-like-particle" else box*apix*0.35)
        rvox = d/2/apix + 3
        mp = f"{args.out}/masks/{sp}.mrc"
        write_mask(mp, box, rvox, apix)
        masks[sp] = mp

    for sp in (args.species if not args.experimental_only else []):
        diam = DIAM.get(sp, 250)
        mask = masks[sp]

        # ---- clean truth reconstruction (for accuracy) ----
        cdir = f"{DATASET}/relion_sta_clean/{sp}"
        clean_map = f"{cdir}/_ppclean/run_it001_class001.mrc"
        if os.path.exists(f"{cdir}/particles.star"):
            if not (args.skip_recon and os.path.exists(clean_map)):
                clean_map = reconstruct(cdir, "particles.star", "_ppclean/run", diam)
        else:
            clean_map = None
        if clean_map and not os.path.exists(clean_map):
            clean_map = None

        gt_avg = {}
        for variant in args.variants:
            projdir = f"{DATASET}/{VARIANT_DIR[variant]}/{sp}"
            star = f"{projdir}/particles.star"
            if not os.path.exists(star):
                print(f"[skip] {variant}/{sp}: no particles.star"); continue
            N = n_particles(star)
            gt = f"{projdir}/{GTMAP}"
            gt_avg[variant] = gt if os.path.exists(gt) else None

            # (1) half-map precision
            h1s, h2s = f"{projdir}/_pph1.star", f"{projdir}/_pph2.star"
            split_star(star, h1s, h2s)
            m1 = f"{projdir}/_ppH1/run_it001_class001.mrc"
            m2 = f"{projdir}/_ppH2/run_it001_class001.mrc"
            if not (args.skip_recon and os.path.exists(m1) and os.path.exists(m2)):
                m1 = reconstruct(projdir, "_pph1.star", "_ppH1/run", diam)
                m2 = reconstruct(projdir, "_pph2.star", "_ppH2/run", diam)
            if m1 and m2 and os.path.exists(m1) and os.path.exists(m2):
                star_pp, r = postprocess(m1, m2, mask, apix, f"{args.out}/pp/half_{variant}_{sp}/pp")
                if star_pp:
                    cols, final = parse_pp_star(star_pp)
                    fsc = pick_fsc(cols, ["rlnFourierShellCorrelationCorrected",
                                          "rlnFourierShellCorrelationMaskedMaps"])
                    ar = cols.get("rlnAngstromResolution")
                    if fsc is not None and ar is not None:
                        r05 = crossing(ar, fsc, 0.5); r143 = crossing(ar, fsc, 0.143)
                        rows.append((f"half-map:{variant}", sp, N, final, "0.143(corr)", r05, r143))
                        curves.append((f"half-map:{variant}", sp, ar, fsc))
                        print(f"half-map {variant:6s} {sp:20s} N={N} FinalRes={final} 0.143={r143:.1f} 0.5={r05:.1f}")
                else:
                    print(f"[warn] postprocess failed half {variant}/{sp}:\n{r.stderr[-400:]}")

            # (2) accuracy vs clean truth
            if clean_map and gt_avg[variant]:
                star_pp, r = postprocess(gt_avg[variant], clean_map, mask, apix,
                                         f"{args.out}/pp/truth_{variant}_{sp}/pp")
                if star_pp:
                    cols, final = parse_pp_star(star_pp)
                    fsc = pick_fsc(cols, ["rlnFourierShellCorrelationMaskedMaps",
                                          "rlnFourierShellCorrelationCorrected"])
                    ar = cols.get("rlnAngstromResolution")
                    if fsc is not None and ar is not None:
                        r05 = crossing(ar, fsc, 0.5); r143 = crossing(ar, fsc, 0.143)
                        rows.append((f"vs-truth:{variant}", sp, N, r05, "0.5(masked)", r05, r143))
                        curves.append((f"vs-truth:{variant}", sp, ar, fsc))
                        print(f"vs-truth {variant:6s} {sp:20s} 0.5={r05:.1f} 0.143={r143:.1f}")

            # cleanup half work dirs (keep small; comment out to inspect)
            for w in ["_pph1.star", "_pph2.star", "_ppH1", "_ppH2"]:
                run(["rm", "-rf", f"{projdir}/{w}"])

        # (3) basic vs faket agreement
        if gt_avg.get("basic") and gt_avg.get("faket"):
            star_pp, r = postprocess(gt_avg["basic"], gt_avg["faket"], mask, apix,
                                     f"{args.out}/pp/cross_basic_faket_{sp}/pp")
            if star_pp:
                cols, final = parse_pp_star(star_pp)
                fsc = pick_fsc(cols, ["rlnFourierShellCorrelationMaskedMaps",
                                      "rlnFourierShellCorrelationCorrected"])
                ar = cols.get("rlnAngstromResolution")
                if fsc is not None and ar is not None:
                    r05 = crossing(ar, fsc, 0.5); r143 = crossing(ar, fsc, 0.143)
                    rows.append(("cross:basic-vs-faket", sp, None, r05, "0.5(masked)", r05, r143))
                    curves.append(("cross:basic-vs-faket", sp, ar, fsc))
                    print(f"cross    basic/faket {sp:20s} 0.5={r05:.1f}")

        if clean_map and args.skip_recon is False:
            run(["rm", "-rf", f"{cdir}/_ppclean"])

    # (4) experimental de-novo -- use RELION's OWN gold-standard FSC from the
    # auto-refine run_model.star (the refine already split the data into halves and
    # computed the FSC; that IS the native RELION resolution). postprocess on those
    # half-maps fails ("FSC never drops below randomize_at_fsc") because auto-refine
    # half-maps are too correlated for post-hoc phase randomization.
    for sp in args.species:
        ename = EXP_NAME.get(sp, sp)
        rdir = refine_dir(ename)
        modstar = f"{EXP_DIR}/{ename}/{rdir}/run_model.star"
        if not os.path.exists(modstar):
            print(f"[skip] experimental {ename}: no {rdir}/run_model.star")
            continue
        ar, fsc, modres = parse_model_star(modstar)
        if ar is not None and fsc is not None:
            r05 = crossing(ar, fsc, 0.5); r143 = crossing(ar, fsc, 0.143)
            sym = rdir.replace("Refine3D", "").lstrip("_") or "C1"
            comp = "experimental" + (f"({sym})" if sym != "C1" else "")
            rows.append((comp, ename, None, modres, "current(0.143)", r05, r143))
            curves.append(("experimental", ename, ar, fsc))
            print(f"exp {sym:>3s} {ename:22s} modelRes={modres} 0.143={r143:.1f} 0.5={r05:.1f}")

    # ---- merge with existing outputs when only recomputing experimental ----
    is_exp = lambda comp: comp.startswith("experimental")
    prev_curve_lines = []
    if args.experimental_only:
        p = f"{args.out}/resolution_summary.csv"
        prev = []
        if os.path.exists(p):
            with open(p) as fh:
                next(fh, None)
                for line in fh:
                    c = line.rstrip("\n").split(",")
                    if len(c) < 7 or is_exp(c[0]):
                        continue
                    prev.append((c[0], c[1], int(c[2]) if c[2] else None,
                                 float(c[3]), c[4], float(c[5]), float(c[6])))
        rows = prev + rows                       # keep sim rows, replace experimental
        pc = f"{args.out}/fsc_curves.csv"
        if os.path.exists(pc):
            with open(pc) as fh:
                next(fh, None)
                prev_curve_lines = [l for l in fh if not is_exp(l.split(",")[0])]

    # ---- write outputs ----
    with open(f"{args.out}/resolution_summary.csv", "w") as f:
        f.write("comparison,species,n_particles,res_primary_A,primary_threshold,res_0.5_A,res_0.143_A\n")
        for comp, sp, n, rp, thr, r05, r143 in rows:
            f.write(f"{comp},{sp},{n if n is not None else ''},"
                    f"{rp:.1f},{thr},{r05:.1f},{r143:.1f}\n")
    with open(f"{args.out}/fsc_curves.csv", "w") as f:
        f.write("comparison,species,resolution_A,fsc\n")
        for line in prev_curve_lines:
            f.write(line if line.endswith("\n") else line + "\n")
        for comp, sp, ar, fsc in curves:
            for a, v in zip(ar, fsc):
                f.write(f"{comp},{sp},{a:.3f},{v:.4f}\n")
    with open(f"{args.out}/report.txt", "w") as f:
        f.write("RELION-native FSC (relion_postprocess) -- separate cross-check of the numpy report\n")
        f.write(f"box={box} apix={apix}A (exp {args.apix_exp}A) -> Nyquist={2*apix:.0f}A\n\n")
        f.write(f"{'comparison':22s} {'species':20s} {'N':>6s} {'primary':>9s} {'thr':>12s} {'0.5(A)':>8s} {'0.143(A)':>9s}\n")
        for comp, sp, n, rp, thr, r05, r143 in rows:
            f.write(f"{comp:22s} {sp:20s} {str(n or '-'):>6s} {rp:9.1f} {thr:>12s} {r05:8.1f} {r143:9.1f}\n")
        f.write("""
NOTES
- FSC here is from relion_postprocess (RELION's masked + phase-randomization
  corrected FSC). This is the AUTHORITATIVE FSC set for the project; the earlier
  in-house numpy FSC dirs (results/fsc_basic_vs_faket, results/resolution) were
  removed because the numpy half-map metric over-reports (pins to Nyquist) where
  RELION's mask/phase-randomization correction gives the honest number.
- half-map:*   PRECISION. primary = RELION FinalResolution (corrected FSC 0.143,
               gold standard). This is the canonical RELION number.
- vs-truth:*   ACCURACY vs the clean-density GT reconstruction. Masked FSC read
               at 0.5 (map-to-model convention); phase-randomization 'corrected'
               FSC is meaningless here (inputs are not true half-maps), so the
               masked FSC + 0.5 crossing is the honest number. primary = 0.5.
               NB: report ACCURACY, not the flattering half-map PRECISION.
- cross:*      basic-vs-faket agreement. Masked FSC @0.5. primary = 0.5.
- experimental de-novo resolution from run_model.star gold-standard FSC. VLP and
  apo-ferritin use their SYMMETRY refines (Refine3D_I / Refine3D_O); the C1 refine
  failed for those symmetric particles. See results/exp_compare for experimental
  half-map (numpy vs RELION) and sim-vs-experimental cross-FSC.
""")
    print(f"\nWrote RELION FSC report -> {args.out}")


if __name__ == "__main__":
    main()

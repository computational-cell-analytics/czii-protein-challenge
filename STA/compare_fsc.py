#!/usr/bin/env python
"""
Fourier Shell Correlation between the two STA variants (basic vs FakET), or
between any two maps. Writes CSV reports + SVG plots. No matplotlib needed.

Runs in the pro-revelio env (numpy + mrcfile).

Default (compare basic vs faket GT-averages for all species):
    micromamba activate pro-revelio
    python compare_fsc.py

Compare two arbitrary maps:
    python compare_fsc.py --map-a A.mrc --map-b B.mrc --label mycompare

Options:
    --species ribosome vlp ...     subset of species (default: all 7)
    --variant-a / --variant-b      basic|faket (default basic vs faket)
    --job GTaverage|Refine3D       which result to compare (default GTaverage)
    --out DIR                      output directory
    --box N                        box size (default 64)
    --apix A                       pixel size (default 10)

Outputs in --out:
    fsc_summary.csv    one row per species: CC, FSC=0.5, FSC=0.143 resolutions
    fsc_curves.csv     full FSC curve (long format) for external plotting
    fsc_all.svg        all curves overlaid
    fsc_<species>.svg  one plot per species
    report.txt         human-readable summary + interpretation notes
"""
import argparse, glob, os
import numpy as np
import mrcfile

DATASET = "/mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_dataset_20260423"
SPECIES_ALL = ["ribosome", "vlp", "beta-galactosidase", "thyroglobulin",
               "apo-ferritin", "beta-amylase", "albumin"]
DIAM = {"vlp": 320, "ribosome": 300, "beta-galactosidase": 180, "thyroglobulin": 250,
        "apo-ferritin": 130, "beta-amylase": 130, "albumin": 100}
VARIANT_DIR = {"basic": "relion_sta", "faket": "relion_sta_faket"}
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

# ---------- FSC math ----------------------------------------------------------
def load(path):
    return mrcfile.open(path, permissive=True).data.astype(np.float32)

def soft_sphere(box, radius_vox, edge=3.0):
    H = box // 2
    zz, yy, xx = np.mgrid[0:box, 0:box, 0:box]
    r = np.sqrt((zz-H)**2 + (yy-H)**2 + (xx-H)**2)
    return np.clip((radius_vox - r)/edge, 0, 1).astype(np.float32)

def fsc_curve(a, b, mask, box):
    H = box // 2
    Fa, Fb = np.fft.fftn(a*mask), np.fft.fftn(b*mask)
    k = np.fft.fftfreq(box); KZ, KY, KX = np.meshgrid(k, k, k, indexing="ij")
    shell = np.clip((np.sqrt(KZ**2+KY**2+KX**2)*box).astype(int), 0, H)
    out = []
    for s in range(1, H):
        m = shell == s
        num = np.real(np.sum(Fa[m]*np.conj(Fb[m])))
        den = np.sqrt(np.sum(np.abs(Fa[m])**2)*np.sum(np.abs(Fb[m])**2))
        out.append(num/den if den > 0 else 0.0)
    return np.array(out)                       # length H-1, shells 1..H-1

def crossing_res(fsc, box, apix, thr):
    """First frequency (linear-interp) where FSC drops below thr -> resolution A."""
    freqs = np.arange(1, len(fsc)+1) / (box*apix)     # cycles/A per shell
    for i, v in enumerate(fsc):
        if v < thr:
            if i == 0:
                return 1.0/freqs[0]
            f0, f1, y0, y1 = freqs[i-1], freqs[i], fsc[i-1], fsc[i]
            fc = f0 + (f1-f0)*(y0-thr)/(y0-y1) if y0 != y1 else f1
            return 1.0/fc
    return 2*apix                                       # never crosses -> Nyquist

def real_cc(a, b, mask):
    a = (a*mask).ravel(); b = (b*mask).ravel()
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def n_particles(star):
    try:
        return sum(1 for l in open(star) if "Subtomograms/" in l)
    except OSError:
        return None

# ---------- minimal SVG line plotter (no deps) --------------------------------
def svg_plot(curves, box, apix, out_path, title):
    """curves: list of (label, color, freqs[1/A], fsc[])."""
    W, Hpx = 820, 520
    L, R, T, B = 96, 210, 52, 70            # wider left margin so y-title + ticks don't collide
    pw, ph = W-L-R, Hpx-T-B
    fmax = 0.5/apix
    ymin, ymax = -0.15, 1.0
    PINK, GREEN = "#d81b8c", "#1a8a1a"
    def X(f): return L + (f/fmax)*pw
    def Y(v): return T + (1 - (v-ymin)/(ymax-ymin))*ph
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
         'font-family="sans-serif" font-size="13">' % (W, Hpx)]
    s.append('<rect width="%d" height="%d" fill="white"/>' % (W, Hpx))
    s.append('<text x="%d" y="28" font-size="16" font-weight="bold">%s</text>' % (L, title))
    # horizontal gridlines: short numeric label at the y-axis; descriptive tag at the RIGHT end
    for yv, num, col, tag in [(1.0,"1.0","#ddd",None), (0.5,"0.5",PINK,"0.5 (agreement)"),
                              (0.143,"0.143",GREEN,"0.143 criterion"), (0.0,"0","#ddd",None)]:
        yy = Y(yv)
        s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-dasharray="4 3"/>'
                 % (L, yy, L+pw, yy, col))
        s.append('<text x="%d" y="%.1f" text-anchor="end" fill="%s">%s</text>'
                 % (L-8, yy+4, col if tag else "#999", num))
        if tag:                              # descriptive tag inside plot, right end, above the line
            s.append('<text x="%d" y="%.1f" text-anchor="end" fill="%s" font-size="11">%s</text>'
                     % (L+pw-6, yy-5, col, tag))
    # x ticks at nice resolutions
    for res in [200, 100, 60, 40, 30, 25, 2*apix]:
        f = 1.0/res
        if f > fmax*1.001: continue
        xx = X(f)
        s.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#eee"/>' % (xx, T, xx, T+ph))
        s.append('<text x="%.1f" y="%d" text-anchor="middle" fill="#333">%d</text>'
                 % (xx, T+ph+18, round(res)))
    s.append('<text x="%d" y="%d" text-anchor="middle">resolution (A)   (left = low res, right = high res)</text>'
             % (L+pw//2, T+ph+46))
    yc = T+ph//2
    s.append('<text x="26" y="%d" text-anchor="middle" transform="rotate(-90 26 %d)">Fourier Shell Correlation</text>'
             % (yc, yc))
    # axes box
    s.append('<rect x="%d" y="%d" width="%d" height="%d" fill="none" stroke="#333"/>' % (L, T, pw, ph))
    # curves + legend
    for i, (label, color, freqs, fsc) in enumerate(curves):
        pts = " ".join("%.1f,%.1f" % (X(f), Y(v)) for f, v in zip(freqs, fsc))
        s.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>' % (pts, color))
        ly = T + 10 + i*22
        s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="3"/>'
                 % (L+pw+18, ly, L+pw+42, ly, color))
        s.append('<text x="%d" y="%d">%s</text>' % (L+pw+48, ly+4, label))
    s.append('</svg>')
    open(out_path, "w").write("\n".join(s))

# ---------- main --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="+", default=SPECIES_ALL)
    ap.add_argument("--variant-a", default="basic")
    ap.add_argument("--variant-b", default="faket")
    ap.add_argument("--job", default="GTaverage", help="GTaverage or Refine3D")
    ap.add_argument("--map-a", help="compare two explicit maps instead of the species sweep")
    ap.add_argument("--map-b")
    ap.add_argument("--label", default="compare")
    ap.add_argument("--box", type=int, default=64)
    ap.add_argument("--apix", type=float, default=10.0)
    ap.add_argument("--out", default="/user/muth9/u12095/czii-protein-challenge/STA/results/fsc_{a}_vs_{b}")
    args = ap.parse_args()

    box, apix = args.box, args.apix
    mapfile = "run_it001_class001.mrc" if args.job == "GTaverage" else "run_class001.mrc"

    # build the list of comparisons
    if args.map_a and args.map_b:
        jobs = [(args.label, args.map_a, args.map_b, box//2 - 4, None)]
        out = args.out.format(a=args.variant_a, b=args.variant_b) if "{" in args.out else args.out
    else:
        out = args.out.format(a=args.variant_a, b=args.variant_b)
        da, db = VARIANT_DIR[args.variant_a], VARIANT_DIR[args.variant_b]
        jobs = []
        for sp in args.species:
            fa = f"{DATASET}/{da}/{sp}/{args.job}/{mapfile}"
            fb = f"{DATASET}/{db}/{sp}/{args.job}/{mapfile}"
            star = f"{DATASET}/{da}/{sp}/particles.star"
            rvox = DIAM.get(sp, box*apix*0.35)/2/apix + 3
            jobs.append((sp, fa, fb, rvox, star))
    os.makedirs(out, exist_ok=True)

    rows, curves_long, plot_curves = [], [], []
    for i, (name, fa, fb, rvox, star) in enumerate(jobs):
        if not (os.path.exists(fa) and os.path.exists(fb)):
            print(f"[skip] {name}: missing {fa if not os.path.exists(fa) else fb}")
            continue
        a, b = load(fa), load(fb)
        mask = soft_sphere(box, rvox)
        fsc = fsc_curve(a, b, mask, box)
        freqs = np.arange(1, len(fsc)+1)/(box*apix)
        cc = real_cc(a, b, mask)
        r05, r0143 = crossing_res(fsc, box, apix, 0.5), crossing_res(fsc, box, apix, 0.143)
        npart = n_particles(star) if star else None
        rows.append((name, npart, rvox*apix, cc, r05, r0143))
        for f, v in zip(freqs, fsc):
            curves_long.append((name, 1.0/f, f, v))
        plot_curves.append((name, OKABE_ITO[i % len(OKABE_ITO)], freqs, fsc))
        # per-species plot
        svg_plot([(name, OKABE_ITO[i % len(OKABE_ITO)], freqs, fsc)], box, apix,
                 f"{out}/fsc_{name}.svg", f"FSC {args.variant_a} vs {args.variant_b}: {name}")
        print(f"{name:20s} CC={cc:.3f}  FSC0.5={r05:.0f}A  FSC0.143={r0143:.0f}A")

    # overlay plot
    if plot_curves:
        svg_plot(plot_curves, box, apix, f"{out}/fsc_all.svg",
                 f"FSC: {args.variant_a} vs {args.variant_b} ({args.job})")

    # summary CSV
    with open(f"{out}/fsc_summary.csv", "w") as f:
        f.write("species,n_particles,mask_radius_A,real_space_CC,FSC_0.5_A,FSC_0.143_A\n")
        for name, npart, mr, cc, r05, r0143 in rows:
            f.write(f"{name},{npart if npart is not None else ''},{mr:.0f},{cc:.4f},{r05:.1f},{r0143:.1f}\n")
    # curves CSV
    with open(f"{out}/fsc_curves.csv", "w") as f:
        f.write("species,resolution_A,spatial_freq_invA,fsc\n")
        for name, res, freq, v in curves_long:
            f.write(f"{name},{res:.2f},{freq:.5f},{v:.4f}\n")
    # report
    with open(f"{out}/report.txt", "w") as f:
        f.write(f"FSC comparison: {args.variant_a} vs {args.variant_b}  (job={args.job})\n")
        f.write(f"box={box}  apix={apix} A  ->  Nyquist={2*apix:.0f} A\n\n")
        f.write(f"{'species':20s} {'N':>6s} {'CC':>7s} {'FSC0.5(A)':>10s} {'FSC0.143(A)':>12s}\n")
        for name, npart, mr, cc, r05, r0143 in rows:
            f.write(f"{name:20s} {str(npart or '-'):>6s} {cc:7.3f} {r05:10.0f} {r0143:12.0f}\n")
        f.write("""
NOTES
- This is a CROSS-FSC between two DIFFERENT reconstructions -> it measures how
  much the two variants AGREE, not the resolution/quality of either one.
- Read the FSC=0.5 crossing as "the resolution to which the two maps agree".
  (The 0.143 criterion is for half-maps of the SAME data, not for this.)
- real_space_CC is a single-number summary over the masked particle region.
- These are GT-angle reconstructions (--skip_align) at 10 A/voxel with weak
  orientational contrast, so absolute resolutions are coarse; the ranking and
  the CC are the meaningful part.
- To instead ask "which variant reconstructs BETTER", run a de-novo auto-refine
  on each and compare each one's own half-map (gold-standard) FSC.
""")
    print(f"\nWrote reports + plots to: {out}")

if __name__ == "__main__":
    main()

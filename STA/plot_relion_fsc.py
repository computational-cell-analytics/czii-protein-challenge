#!/usr/bin/env python
"""
Render SVG FSC plots for the RELION results, reusing the SAME svg_plot() the
numpy pipeline used (compare_fsc.py) so the figures match the numpy ones.

Reads the curves.csv written by relion_fsc_report.py and exp_compare_fsc.py;
writes SVGs into results/relion_fsc/plots/ and results/exp_compare/plots/.

    micromamba activate pro-revelio
    python plot_relion_fsc.py
"""
import os, csv, collections
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_fsc import svg_plot, OKABE_ITO

RELION_FSC = "/user/muth9/u12095/czii-protein-challenge/STA/results/relion_fsc"
EXP_CMP = "/user/muth9/u12095/czii-protein-challenge/STA/results/exp_compare"
BOX, APIX = 64, 10.0
FMAX = 0.5 / APIX

# colour scheme matching resolution_report.py
COL = {"half-map:basic": "#0072B2", "half-map:faket": "#E69F00",
       "vs-truth:basic": "#009E73", "vs-truth:faket": "#D55E00",
       "relion/basic-vs-exp": "#0072B2", "relion/faket-vs-exp": "#D55E00",
       "numpy/exp-halfmap": "#000000"}
LABEL = {"half-map:basic": "basic half-map", "half-map:faket": "faket half-map",
         "vs-truth:basic": "basic vs truth", "vs-truth:faket": "faket vs truth",
         "relion/basic-vs-exp": "basic vs exp", "relion/faket-vs-exp": "faket vs exp",
         "numpy/exp-halfmap": "exp half-map (numpy)"}


def to_curve(res, fsc):
    """(resolution_A[], fsc[]) -> (freqs_invA[], fsc[]) sorted by freq, in range."""
    res = np.asarray(res, float); fsc = np.asarray(fsc, float)
    ok = np.isfinite(res) & (res > 0)
    f = 1.0 / res[ok]; v = fsc[ok]
    keep = f <= FMAX * 1.02
    f, v = f[keep], v[keep]
    order = np.argsort(f)
    return f[order], v[order]


def load(path, keycols):
    """Return {key_tuple: {species: (freqs, fsc)}} keyed by joined keycols."""
    rows = collections.defaultdict(lambda: collections.defaultdict(lambda: ([], [])))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = "/".join(r[c] for c in keycols)
            sp = r["species"]
            rows[key][sp][0].append(float(r["resolution_A"]))
            rows[key][sp][1].append(float(r["fsc"]))
    return rows


def main():
    os.makedirs(f"{RELION_FSC}/plots", exist_ok=True)
    os.makedirs(f"{EXP_CMP}/plots", exist_ok=True)

    # ---------- relion_fsc ----------
    rf = load(f"{RELION_FSC}/fsc_curves.csv", ["comparison"])
    species = sorted({sp for comp in rf.values() for sp in comp})

    # (1) per-species quality: 4 curves (half-map + vs-truth, basic & faket)
    for sp in species:
        curves = []
        for comp in ["half-map:basic", "half-map:faket", "vs-truth:basic", "vs-truth:faket"]:
            if comp in rf and sp in rf[comp]:
                f, v = to_curve(*rf[comp][sp])
                if len(f):
                    curves.append((LABEL[comp], COL[comp], f, v))
        if curves:
            svg_plot(curves, BOX, APIX, f"{RELION_FSC}/plots/fsc_quality_{sp}.svg",
                     f"RELION FSC (postprocess): {sp}")

    # (2) basic-vs-faket cross: per species + overlay
    cross = rf.get("cross:basic-vs-faket", {})
    overlay = []
    for i, sp in enumerate(sorted(cross)):
        f, v = to_curve(*cross[sp])
        if not len(f):
            continue
        col = OKABE_ITO[i % len(OKABE_ITO)]
        svg_plot([(sp, col, f, v)], BOX, APIX,
                 f"{RELION_FSC}/plots/fsc_cross_{sp}.svg",
                 f"RELION FSC basic-vs-faket: {sp}")
        overlay.append((sp, col, f, v))
    if overlay:
        svg_plot(overlay, BOX, APIX, f"{RELION_FSC}/plots/fsc_cross_all.svg",
                 "RELION FSC basic-vs-faket (all species)")

    # (2b) half-map (gold-standard) overlay across species, one per variant
    #      -- the basic/faket analogue of fsc_experimental_all.svg
    for comp, tag, title in [("half-map:basic", "fsc_halfmap_basic_all",
                              "RELION gold-standard half-map FSC, basic (all species)"),
                             ("half-map:faket", "fsc_halfmap_faket_all",
                              "RELION gold-standard half-map FSC, faket (all species)")]:
        overlay = []
        for i, sp in enumerate(sorted(rf.get(comp, {}))):
            f, v = to_curve(*rf[comp][sp])
            if len(f):
                overlay.append((sp, OKABE_ITO[i % len(OKABE_ITO)], f, v))
        if overlay:
            svg_plot(overlay, BOX, APIX, f"{RELION_FSC}/plots/{tag}.svg", title)

    # (3) experimental gold-standard FSC: per species + overlay
    exp = rf.get("experimental", {})
    overlay = []
    for i, sp in enumerate(sorted(exp)):
        f, v = to_curve(*exp[sp])
        if not len(f):
            continue
        col = OKABE_ITO[i % len(OKABE_ITO)]
        svg_plot([(sp, col, f, v)], BOX, APIX,
                 f"{RELION_FSC}/plots/fsc_experimental_{sp}.svg",
                 f"RELION gold-standard FSC (de-novo): {sp}")
        overlay.append((sp, col, f, v))
    if overlay:
        svg_plot(overlay, BOX, APIX, f"{RELION_FSC}/plots/fsc_experimental_all.svg",
                 "RELION gold-standard FSC, experimental de-novo (all species)")

    # ---------- exp_compare: per-species sim-vs-exp + exp half-map ----------
    ec = load(f"{EXP_CMP}/curves.csv", ["method", "comparison"])
    species_e = sorted({sp for comp in ec.values() for sp in comp})
    for sp in species_e:
        curves = []
        for comp in ["relion/basic-vs-exp", "relion/faket-vs-exp", "numpy/exp-halfmap"]:
            if comp in ec and sp in ec[comp]:
                f, v = to_curve(*ec[comp][sp])
                if len(f):
                    curves.append((LABEL[comp], COL[comp], f, v))
        if curves:
            svg_plot(curves, BOX, APIX, f"{EXP_CMP}/plots/fsc_{sp}.svg",
                     f"Experimental FSC: {sp}")

    # all-species overlays: how similar is the real map to each sim variant
    for comp, tag, title in [("relion/faket-vs-exp", "faket_vs_exp_all",
                              "RELION FSC faket-vs-experimental (all species)"),
                             ("relion/basic-vs-exp", "basic_vs_exp_all",
                              "RELION FSC basic-vs-experimental (all species)")]:
        overlay = []
        for i, sp in enumerate(sorted(ec.get(comp, {}))):
            f, v = to_curve(*ec[comp][sp])
            if len(f):
                overlay.append((sp, OKABE_ITO[i % len(OKABE_ITO)], f, v))
        if overlay:
            svg_plot(overlay, BOX, APIX, f"{EXP_CMP}/plots/{tag}.svg", title)

    n1 = len(os.listdir(f"{RELION_FSC}/plots"))
    n2 = len(os.listdir(f"{EXP_CMP}/plots"))
    print(f"Wrote {n1} SVGs -> {RELION_FSC}/plots/")
    print(f"Wrote {n2} SVGs -> {EXP_CMP}/plots/")


if __name__ == "__main__":
    main()

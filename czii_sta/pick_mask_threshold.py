#!/usr/bin/env python
"""
Choose a binarisation threshold for relion_mask_create from the refined map.

Otsu on the voxel histogram: the map is bimodal (solvent vs particle), so the
threshold that minimises within-class variance separates them without needing any
external knowledge of the specimen. The expected diameter is used only as a sanity
check, never to set the value.

Prints the threshold on stdout; diagnostics go to stderr so the caller can do
  THRESHOLD=$(pick_mask_threshold.py --i map.mrc --expect-diameter 300)
"""
import argparse, sys
import numpy as np, mrcfile


def otsu(v, nbins=512):
    hist, edges = np.histogram(v, bins=nbins)
    centres = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(hist); wb = w / w[-1]
    m = np.cumsum(hist * centres)
    mu_b = np.divide(m, w, out=np.zeros_like(m, float), where=w > 0)
    mu_f = np.divide(m[-1] - m, w[-1] - w, out=np.zeros_like(m, float), where=(w[-1] - w) > 0)
    var = wb * (1 - wb) * (mu_b - mu_f) ** 2
    return float(centres[int(np.nanargmax(var))])


def diameter_of(vol_vox, apix):
    """Diameter of the sphere with the same volume, in Angstrom."""
    return 2.0 * ((3.0 * vol_vox * apix ** 3) / (4.0 * np.pi)) ** (1.0 / 3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i", required=True)
    ap.add_argument("--expect-diameter", type=float, default=None,
                    help="approximate particle diameter in A, for the sanity check only")
    args = ap.parse_args()

    with mrcfile.open(args.i, permissive=True) as m:
        v = m.data.astype(np.float32).copy()
        apix = float(m.voxel_size.x)

    t = otsu(v)
    n = int((v > t).sum())
    d = diameter_of(n, apix)
    e = sys.stderr

    print(f"map        : {args.i}", file=e)
    print(f"box {v.shape}  pixel {apix:.3f} A", file=e)
    print(f"range      : {v.min():+.4f} .. {v.max():+.4f}   mean {v.mean():+.4f}  std {v.std():.4f}", file=e)
    print(f"\notsu threshold : {t:.5f}", file=e)
    print(f"  voxels above : {n}  ({100.0*n/v.size:.1f}% of the box)", file=e)
    print(f"  equivalent sphere diameter : {d:.0f} A", file=e)

    print("\nfor comparison (not used):", file=e)
    for k in (1, 2, 3, 4):
        tk = v.mean() + k * v.std()
        nk = int((v > tk).sum())
        print(f"  mean+{k}sigma = {tk:8.5f} -> {nk:8d} vox, sphere {diameter_of(nk, apix):5.0f} A", file=e)

    if args.expect_diameter:
        r = d / args.expect_diameter
        print(f"\nsanity check vs expected {args.expect_diameter:.0f} A: ratio {r:.2f}", file=e)
        if not 0.5 <= r <= 1.6:
            print("  ** WARNING: enclosed volume is far from the expected particle size.", file=e)
            print("  ** Inspect the map and set the threshold by hand before trusting the mask.", file=e)
        else:
            print("  plausible.", file=e)

    print(f"{t:.5f}")          # stdout: the value itself


if __name__ == "__main__":
    main()

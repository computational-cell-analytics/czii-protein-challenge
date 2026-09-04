#!/usr/bin/env python
"""
Measure the offset between a map and a reference mask along one axis.

  python measure_map_offset.py --map Refine3D/run_class001.mrc \
                               --mask masks/mask_align_bin4.mrc [--star run_data.star]

Reports FOUR estimators and flags when they disagree. Disagreement is the useful
signal: it means the map holds more than one feature along the axis (e.g. HIV capsid
AND matrix) while the mask covers one, so any estimator using total density is pulled
toward whichever layer carries more mass. The peak estimator avoids that, but only
when the map has ONE clear maximum inside the mask; if the peak lands against the
support edge the script falls back to the in-support centroid. Either way the value is
an estimate -- confirm it by reconstructing and re-measuring, and expect to iterate.
"""
import argparse
import numpy as np
import mrcfile

AXES = {"x": 2, "y": 1, "z": 0}


def load(path):
    with mrcfile.open(path, permissive=True) as m:
        return m.data.astype(np.float32).copy(), float(m.voxel_size.x)


def profile(vol, axis):
    return vol.mean(axis=tuple(i for i in range(vol.ndim) if i != axis))


def parabolic(y3):
    d = y3[0] - 2 * y3[1] + y3[2]
    return 0.0 if d == 0 else 0.5 * (y3[0] - y3[2]) / d


def shift1d(x, s):
    out = np.roll(x, s)
    if s > 0:
        out[:s] = 0
    elif s < 0:
        out[s:] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--axis", default="z", choices=list(AXES))
    ap.add_argument("--range", type=int, default=20, help="max shift to scan, px")
    ap.add_argument("--star", help="print a ready relion_star_handler command for this STAR")
    args = ap.parse_args()

    vol, apix = load(args.map)
    mask, apix_m = load(args.mask)
    if vol.shape != mask.shape:
        raise SystemExit(f"box mismatch: {vol.shape} vs {mask.shape}")
    if apix and apix_m and abs(apix - apix_m) > 1e-3:
        print(f"WARNING: pixel size differs ({apix:.3f} vs {apix_m:.3f} A)")

    ax = AXES[args.axis]
    a, b = profile(vol, ax), profile(mask, ax)
    n = np.arange(len(a))

    sup = np.where(b > 0.01 * b.max())[0]
    lo, hi = int(sup.min()), int(sup.max())
    c_mask = float(np.average(n, weights=b))

    # Primary: strongest density peak inside the mask's support, sub-pixel refined.
    # Immune to material outside the mask, unlike anything integrating total density.
    seg = a[lo:hi + 1]
    k = int(seg.argmax()) + lo
    clipped = (k - lo) < 3 or (hi - k) < 3   # peak against the support edge
    k = min(max(k, 1), len(a) - 2)
    peak = k + parabolic(a[k - 1:k + 2])
    est = {"peak-in-mask": c_mask - peak}

    # Secondary estimators, kept to expose disagreement.
    est["centroid in support"] = c_mask - float(
        np.average(n[lo:hi + 1], weights=np.clip(a[lo:hi + 1], 0, None)))

    shifts = np.arange(-args.range, args.range + 1)
    ov = np.array([float((shift1d(a, int(s)) * b).sum()) for s in shifts])
    j = int(ov.argmax())
    est["mask-weighted overlap"] = shifts[j] + (
        parabolic(ov[j - 1:j + 2]) if 0 < j < len(ov) - 1 else 0)

    ac, bc = a - a.mean(), b - b.mean()
    cc = np.array([float(shift1d(ac, int(s)) @ bc /
                        (np.linalg.norm(shift1d(ac, int(s))) * np.linalg.norm(bc) + 1e-12))
                   for s in shifts])
    j = int(cc.argmax())
    est["profile NCC"] = shifts[j] + (parabolic(cc[j - 1:j + 2]) if 0 < j < len(cc) - 1 else 0)

    print(f"\nmap  : {args.map}\nmask : {args.mask}")
    print(f"box {vol.shape}  pixel {apix:.3f} A  axis {args.axis}")
    print(f"mask support {lo}..{hi}, centre {c_mask:.2f}; map peak in support {peak:.2f}\n")

    for name, v in est.items():
        line = f"  {name:24s} {v:+7.2f} px"
        if apix:
            line += f"  ({v * apix:+7.2f} A)"
        print(line)

    # A peak sitting on the support edge means the estimator could not see the real
    # maximum; and a multi-humped map (capsid + matrix) has no single meaningful peak.
    # In both cases the in-support centroid is the more honest summary.
    primary_key = "centroid in support" if clipped else "peak-in-mask"
    primary = est[primary_key]
    if clipped:
        print("\n  NOTE: map peak sits on the mask-support edge -- the real maximum is")
        print("        outside the support. Falling back to the centroid estimator.")
    spread = max(est.values()) - min(est.values())
    print(f"\n  spread across estimators: {spread:.2f} px")
    if spread > 2:
        print("  -> estimators disagree: the map holds features the mask does not cover.")
        print(f"     Using '{primary_key}'. Treat the value as approximate and confirm")
        print("     by reconstructing and re-measuring.")

    print(f"\n  offset ({primary_key}) = {primary:+.2f} px along {args.axis}"
          + (f" = {primary * apix:+.2f} A" if apix else ""))
    print("  Sign convention vs relion_star_handler is NOT verified here — if the")
    print("  reconstruction moves the wrong way, negate it.")

    print(f"\n  map profile (>= mask support marked '|'):")
    rng = np.ptp(a) + 1e-12
    for i in range(0, len(a), 2):
        bar = "#" * int(30 * (a[i] - a.min()) / rng)
        mk = "|" if lo <= i <= hi else " "
        tag = "  <- peak" if abs(i - peak) < 1 else ""
        print(f"    {i:3d} {mk} {bar}{tag}")

    if args.star:
        out = args.star.replace(".star", f"_z{primary:.2f}.star")
        print(f"\n  relion_star_handler --i {args.star} \\\n"
              f"      --o {out} --center --center_{args.axis.upper()} {primary:.2f}")


if __name__ == "__main__":
    main()

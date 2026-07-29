#!/usr/bin/env python
"""
Build low-pass-filtered PDB-model references for STA, one per species, from the
simulation's PDB-derived 10 A density templates. Writes initial_ref.mrc into each
species folder of a RELION project (overwriting the average-of-locations ref).

    micromamba activate pro-revelio
    python build_reference.py --project <proj_root> [--res 50] [--box 64]

Default project = the experimental one. Use --project <dataset>/relion_sta[_faket]
to give the sim runs the same PDB reference (apples-to-apples de-novo comparison).
"""
import argparse, os, glob
import numpy as np, mrcfile

TEMPLATES = "/user/muth9/u12095/simulation/polnet-synaptic/data/default/templates/mrcs_10A"
CZII = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge"
# species name (as used in each project's folders) -> template PDB code
CODE = {"ribosome": "6qzp", "virus-like-particle": "6n4v", "vlp": "6n4v",
        "beta-galactosidase": "6drv", "thyroglobulin": "7n4y",
        "apo-ferritin": "8cpv", "beta-amylase": "1fa2", "albumin": "8vaf"}

def pad_or_crop(v, box):
    out = np.zeros((box, box, box), np.float32)
    s = v.shape[0]
    if s >= box:                         # crop centered
        o = (s - box)//2
        out[:] = v[o:o+box, o:o+box, o:o+box]
    else:                                # pad centered
        o = (box - s)//2
        out[o:o+s, o:o+s, o:o+s] = v
    return out

def lowpass(v, apix, res):
    N = v.shape[0]
    k = np.fft.fftfreq(N); KZ, KY, KX = np.meshgrid(k, k, k, indexing="ij")
    kr = np.sqrt(KZ**2 + KY**2 + KX**2)          # cycles/voxel
    kc = apix/res                                # cutoff (cycles/voxel) for resolution `res`
    w = np.exp(-(kr/kc)**2)                       # soft Gaussian low-pass
    return np.real(np.fft.ifftn(np.fft.fftn(v)*w)).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="/mnt/vast-nhr/projects/nim00007/simulated_cryoET/czii_dataset_20260423/relion_sta_experimental")
    ap.add_argument("--species", nargs="+", default=None, help="default: all species dirs in project")
    ap.add_argument("--res", type=float, default=50.0, help="low-pass resolution (A)")
    ap.add_argument("--box", type=int, default=64)
    ap.add_argument("--apix", type=float, default=10.012)
    args = ap.parse_args()

    sp_dirs = args.species or [os.path.basename(p) for p in glob.glob(f"{args.project}/*")
                              if os.path.isdir(p) and os.path.basename(p) in CODE]
    for sp in sp_dirs:
        code = CODE.get(sp)
        tmpl = f"{TEMPLATES}/{code}.mrc"
        if not (code and os.path.exists(tmpl)):
            print(f"[skip] {sp}: no template ({code})"); continue
        v = mrcfile.open(tmpl, permissive=True).data.astype(np.float32)   # protein bright, 10 A
        v = pad_or_crop(v, args.box)
        v = lowpass(v, args.apix, args.res)
        v = (v - v.mean())/(v.std() + 1e-9)                               # keep protein bright
        outdir = f"{args.project}/{sp}"
        os.makedirs(outdir, exist_ok=True)
        with mrcfile.new(f"{outdir}/initial_ref.mrc", overwrite=True) as m:
            m.set_data(v.astype(np.float32)); m.voxel_size = args.apix
        print(f"{sp:22s} <- {code}.mrc  lowpass {args.res:.0f}A  -> {outdir}/initial_ref.mrc")

if __name__ == "__main__":
    main()

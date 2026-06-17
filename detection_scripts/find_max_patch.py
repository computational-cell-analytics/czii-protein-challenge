"""Empirically find the largest training patch shape (z >= 64) that fits on the GPU.

Mirrors the model used in detection_scripts/train_detection.py:
    AnisotropicUNet, scale_factors=[[1,2,2],[2,2,2],[2,2,2],[2,2,2]],
    in_channels=1, out_channels=5, initial_features=32, gain=2.

The binary-search-over-an-OOM-signal idea is taken from torch_em/util/memory.py.
That file searches with a *forward pass under no_grad*, which reflects INFERENCE
memory only. Training needs far more (activations kept for backward, gradients,
Adam moments, AMP). So we measure two `fits` functions:
    - forward-only  (== what torch_em/util/memory.py does)
    - full training step (forward + loss + backward + optimizer step, mixed precision)
and report the max x/y for each, keeping z fixed.
"""
import gc
import torch
import torch.nn as nn
from torch_em.model import AnisotropicUNet

DIV_Z, DIV_XY = 8, 16          # divisibility of the U-Net (z%8==0, xy%16==0)
IN_CH, OUT_CH = 1, 5


def build_model():
    return AnisotropicUNet(
        scale_factors=[[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        in_channels=IN_CH, out_channels=OUT_CH, initial_features=32, gain=2,
        final_activation=None,
    )


def _is_oom(exc):
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom is not None and isinstance(exc, oom):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def make_fits(model, device, batch_size, z, mode):
    """Return fits(k): does patch (z, DIV_XY*k, DIV_XY*k) fit, in the given mode?"""
    loss_fn = nn.MSELoss()

    def fits(k):
        xy = DIV_XY * k
        inp = tgt = out = opt = None
        try:
            torch.cuda.reset_peak_memory_stats(device)
            inp = torch.empty((batch_size, IN_CH, z, xy, xy), device=device).normal_()
            if mode == "forward":
                model.eval()
                with torch.no_grad():
                    out = model(inp)
            else:  # full training step
                model.train()
                opt = torch.optim.Adam(model.parameters(), lr=1e-4)
                scaler = torch.cuda.amp.GradScaler()
                tgt = torch.empty((batch_size, OUT_CH, z, xy, xy), device=device).normal_()
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    out = model(inp)
                    loss = loss_fn(out, tgt)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            torch.cuda.synchronize(device)
            return True
        except Exception as exc:
            if _is_oom(exc):
                return False
            raise
        finally:
            del inp, tgt, out, opt
            _cleanup()

    return fits


def search_max_k(fits, upper=128):
    """Largest k in [1, upper] with fits(k) True (exponential bracket + binary search)."""
    if not fits(1):
        return 0
    last_ok, cand = 1, 2
    while cand <= upper and fits(cand):
        last_ok, cand = cand, cand * 2
    if cand > upper:
        return upper if fits(upper) else last_ok  # hit the bound
    lo, hi = last_ok, cand
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {name} | total memory: {total:.1f} GiB")
    print(f"divisibility: z%{DIV_Z}==0, xy%{DIV_XY}==0; z fixed >= 64\n")

    model = build_model().to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model params: {nparams/1e6:.2f} M\n")

    Z = 64
    for bs in (1, 2):
        for mode in ("forward", "training"):
            fits = make_fits(model, dev, bs, Z, mode)
            k = search_max_k(fits)
            if k == 0:
                print(f"batch={bs:<2} z={Z} mode={mode:<8} -> does NOT fit even at xy=16")
            else:
                xy = DIV_XY * k
                print(f"batch={bs:<2} z={Z} mode={mode:<8} -> max patch = "
                      f"[{Z}, {xy}, {xy}]  (xy multiple k={k})")
    print()

    # Also: with the training-config batch=2, how much z headroom at the current xy=256?
    print("Extra: training step, batch=2, fixed xy=256, growing z (mult of 8):")
    for z in (64, 72, 80, 96, 128):
        fits = make_fits(model, dev, 2, z, "training")
        ok = fits(256 // DIV_XY)  # k for xy=256
        print(f"  z={z:<4} xy=256 -> {'fits' if ok else 'OOM'}")


if __name__ == "__main__":
    main()

"""Cryo-ET corruption operators for building contrastive positive pairs.

Two views of the same crop differ only in nuisance factors (noise realization, residual CTF,
dose, missing wedge, ice gradient, crop jitter), so class identity is preserved by
construction. Not generic vision augmentations -- everything here has a cryo-ET meaning.

Axes are (z, y, x): beam along z, tilt axis along y.
"""

import math
import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .cryoET_augmentation import random_flip, random_rot90, random_rescale, random_intensity


_WAVELENGTH_ANGSTROM = {80: 0.0418, 120: 0.0335, 200: 0.0251, 300: 0.0197}

_FREQ_CACHE: Dict[tuple, tuple] = {}


def _freq_grids(shape, voxel_size, device):
    """Radial frequency (1/A) plus |kz| and |kx| for an rfftn of `shape`."""
    key = (tuple(shape), float(voxel_size), str(device))
    cached = _FREQ_CACHE.get(key)
    if cached is not None:
        return cached

    d, h, w = shape
    kz = torch.fft.fftfreq(d, d=voxel_size, device=device).view(-1, 1, 1)
    ky = torch.fft.fftfreq(h, d=voxel_size, device=device).view(1, -1, 1)
    kx = torch.fft.rfftfreq(w, d=voxel_size, device=device).view(1, 1, -1)

    k = torch.sqrt(kz ** 2 + ky ** 2 + kx ** 2)
    grids = (k, kz.abs().expand(d, h, kx.shape[-1]), kx.abs().expand(d, h, kx.shape[-1]))
    _FREQ_CACHE[key] = grids
    return grids


def _apply_fourier_filter(volume, filt):
    spatial = volume.shape[1:]
    ft = torch.fft.rfftn(volume.float(), dim=(1, 2, 3)) * filt
    return torch.fft.irfftn(ft, s=spatial, dim=(1, 2, 3))


def add_noise_at_snr(volume, snr_db: float):
    """Add white Gaussian noise at a prescribed SNR.

    Signal power is the crop variance, which is background-dominated -- so `snr_db` is a
    relative severity knob, comparable across crops because they are normalized first.
    """
    signal_power = volume.float().var()
    noise_power = signal_power / (10 ** (snr_db / 10))
    return volume + torch.randn_like(volume) * torch.sqrt(noise_power.clamp_min(1e-12))


def add_correlated_noise(volume, snr_db: float, sigma: float = 1.5):
    """Low-pass filtered noise; closer to reconstruction artefacts than white noise."""
    noise = torch.randn_like(volume)
    size = int(2 * math.ceil(2 * sigma) + 1)
    coords = torch.arange(size, device=volume.device, dtype=torch.float32) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
    kernel = (kernel / kernel.sum()).view(1, 1, size, size, size)
    noise = F.conv3d(noise.unsqueeze(0), kernel, padding=size // 2).squeeze(0)

    # Filtering removes variance, so rescale to the requested SNR afterwards.
    target_power = volume.float().var() / (10 ** (snr_db / 10))
    noise = noise * torch.sqrt(target_power.clamp_min(1e-12) / noise.float().var().clamp_min(1e-12))
    return volume + noise


def apply_ctf(
    volume,
    defocus_um: float,
    b_factor: float = 0.0,
    voxel_size: float = 10.0,
    voltage_kv: int = 300,
    cs_mm: float = 2.7,
    amplitude_contrast: float = 0.07,
    phase_flip: bool = False,
):
    """Modulate by a CTF at the given defocus, with an exp(-B k^2 / 4) envelope.

    The tomograms are already CTF-affected, so this models the residual of an imperfect
    correction. Only |CTF| is applied by default: phases stay intact and the physically
    meaningful part -- information loss at the CTF zeros -- is reproduced. `phase_flip`
    applies the signed CTF instead.
    """
    lam = _WAVELENGTH_ANGSTROM.get(voltage_kv, 0.0197)
    cs = cs_mm * 1e7
    defocus = defocus_um * 1e4

    k, _, _ = _freq_grids(volume.shape[1:], voxel_size, volume.device)
    k2 = k ** 2

    chi = math.pi * lam * defocus * k2 - 0.5 * math.pi * cs * (lam ** 3) * (k2 ** 2)
    w2 = amplitude_contrast
    w1 = math.sqrt(max(0.0, 1.0 - w2 ** 2))
    ctf = w1 * torch.sin(chi) + w2 * torch.cos(chi)
    if not phase_flip:
        ctf = ctf.abs()
    if b_factor > 0:
        ctf = ctf * torch.exp(-b_factor * k2 / 4.0)

    return _apply_fourier_filter(volume, ctf)


def apply_dose_envelope(volume, b_factor: float, voxel_size: float = 10.0):
    """Dose / resolution variation: exp(-B k^2 / 4) without CTF oscillations."""
    k, _, _ = _freq_grids(volume.shape[1:], voxel_size, volume.device)
    return _apply_fourier_filter(volume, torch.exp(-b_factor * (k ** 2) / 4.0))


def apply_missing_wedge(volume, tilt_max_deg: float = 60.0, soft_edge_deg: float = 3.0):
    """Zero the Fourier wedge a +/- `tilt_max_deg` tilt series never samples.

    A projection at tilt alpha contributes the central slice at angle alpha in the (kx, kz)
    plane, so frequencies within ``90 - tilt_max`` of the kz axis are never measured. Applying
    an extra wedge models a shorter tilt range than the data was acquired with.
    """
    _, kz_abs, kx_abs = _freq_grids(volume.shape[1:], 1.0, volume.device)

    angle = torch.atan2(kx_abs, kz_abs.clamp_min(1e-12))
    cutoff = math.radians(90.0 - tilt_max_deg)
    soft = math.radians(max(soft_edge_deg, 1e-3))

    # 0 inside the wedge, 1 outside; soft transition to avoid ringing.
    return _apply_fourier_filter(volume, torch.clamp((angle - cutoff) / soft, 0.0, 1.0))


def add_ice_gradient(volume, amplitude_fraction: float = 0.1):
    """Smooth linear ramp: local ice-thickness / background variation."""
    d, h, w = volume.shape[1:]
    ramps = [torch.linspace(-0.5, 0.5, n, device=volume.device) for n in (d, h, w)]
    direction = torch.randn(3, device=volume.device)
    direction = direction / direction.norm().clamp_min(1e-8)

    grad = (
        direction[0] * ramps[0].view(-1, 1, 1)
        + direction[1] * ramps[1].view(1, -1, 1)
        + direction[2] * ramps[2].view(1, 1, -1)
    )
    return volume + grad.unsqueeze(0) * volume.float().std() * amplitude_fraction


def random_translation_jitter(volume, max_shift: int = 2):
    """Shift by up to `max_shift` voxels per axis; models crop-centre uncertainty."""
    if max_shift <= 0:
        return volume
    shifts = [random.randint(-max_shift, max_shift) for _ in range(3)]
    if all(s == 0 for s in shifts):
        return volume

    # Replicate-pad first so nothing wraps around.
    padded = F.pad(volume.unsqueeze(0), [max_shift] * 6, mode="replicate").squeeze(0)
    d, h, w = volume.shape[1:]
    z0, y0, x0 = (max_shift + s for s in shifts)
    return padded[:, z0:z0 + d, y0:y0 + h, x0:x0 + w]


class NoiseAwareViewAugment:
    """Generate one corrupted view and report how hard the corruption was.

    Returns ``(view, snr_db)``; ``snr_db`` is ``inf`` when no noise was added and drives the
    per-pair weight in the contrastive loss (see :func:`snr_positive_weight`).

    Args:
        severity (call arg): 0 samples only the mild end of the SNR range, 1 the full range.
        snr_db_range / snr_db_range_mild: SNR bands at severity 1 and 0.
        jitter_voxels: Maximum crop-centre shift per axis.
        renormalize: Re-standardize after corruption -- otherwise global mean/std is a trivial
            shortcut for matching two views of the same crop.
    """

    def __init__(
        self,
        snr_db_range: Tuple[float, float] = (-6.0, 12.0),
        snr_db_range_mild: Tuple[float, float] = (3.0, 12.0),
        correlated_noise_prob: float = 0.5,
        rotate: bool = True,
        flip: bool = True,
        jitter_voxels: int = 2,
        p_noise: float = 1.0,
        p_ctf: float = 0.4,
        p_dose: float = 0.4,
        p_wedge: float = 0.3,
        p_ice: float = 0.3,
        p_intensity: float = 0.5,
        p_rescale: float = 0.2,
        defocus_um_range: Tuple[float, float] = (1.0, 5.0),
        b_factor_range: Tuple[float, float] = (0.0, 1500.0),
        tilt_max_deg_range: Tuple[float, float] = (45.0, 70.0),
        rescale_range: Tuple[float, float] = (0.95, 1.05),
        voxel_size: float = 10.0,
        voltage_kv: int = 300,
        renormalize: bool = True,
    ):
        self.snr_db_range = snr_db_range
        self.snr_db_range_mild = snr_db_range_mild
        self.correlated_noise_prob = correlated_noise_prob
        self.rotate = rotate
        self.flip = flip
        self.jitter_voxels = jitter_voxels
        self.p_noise = p_noise
        self.p_ctf = p_ctf
        self.p_dose = p_dose
        self.p_wedge = p_wedge
        self.p_ice = p_ice
        self.p_intensity = p_intensity
        self.p_rescale = p_rescale
        self.defocus_um_range = defocus_um_range
        self.b_factor_range = b_factor_range
        self.tilt_max_deg_range = tilt_max_deg_range
        self.rescale_range = rescale_range
        self.voxel_size = voxel_size
        self.voltage_kv = voltage_kv
        self.renormalize = renormalize

    def _sample_snr_db(self, severity: float) -> float:
        severity = float(min(max(severity, 0.0), 1.0))
        low = self.snr_db_range_mild[0] + severity * (self.snr_db_range[0] - self.snr_db_range_mild[0])
        high = self.snr_db_range_mild[1] + severity * (self.snr_db_range[1] - self.snr_db_range_mild[1])
        if high < low:
            low, high = high, low
        return random.uniform(low, high)

    def __call__(self, volume: torch.Tensor, severity: float = 1.0,
                 force_snr_db: Optional[float] = None) -> Tuple[torch.Tensor, float]:
        if not isinstance(volume, torch.Tensor):
            volume = torch.as_tensor(volume, dtype=torch.float32)
        volume = volume.float()

        # Geometry first, so the Fourier operators act on the final sampling grid.
        # In-plane rot90 is a rotation about z and leaves the wedge geometry intact.
        volume = random_translation_jitter(volume, self.jitter_voxels)
        if self.rotate:
            volume = random_rot90(volume)
        if self.flip:
            volume = random_flip(volume)
        if random.random() < self.p_rescale:
            volume = random_rescale(volume, self.rescale_range)

        if random.random() < self.p_wedge:
            volume = apply_missing_wedge(volume, random.uniform(*self.tilt_max_deg_range))
        if random.random() < self.p_ctf:
            volume = apply_ctf(
                volume,
                defocus_um=random.uniform(*self.defocus_um_range),
                b_factor=random.uniform(*self.b_factor_range) if random.random() < 0.5 else 0.0,
                voxel_size=self.voxel_size,
                voltage_kv=self.voltage_kv,
            )
        elif random.random() < self.p_dose:
            volume = apply_dose_envelope(volume, random.uniform(*self.b_factor_range), self.voxel_size)

        # Noise last: it must not be shaped by the filters above.
        snr_db = float("inf")
        if force_snr_db is not None or random.random() < self.p_noise:
            snr_db = force_snr_db if force_snr_db is not None else self._sample_snr_db(severity)
            if random.random() < self.correlated_noise_prob:
                volume = add_correlated_noise(volume, snr_db)
            else:
                volume = add_noise_at_snr(volume, snr_db)

        if random.random() < self.p_ice:
            volume = add_ice_gradient(volume)
        if random.random() < self.p_intensity:
            volume = random_intensity(volume)

        if self.renormalize:
            volume = (volume - volume.mean()) / (volume.std() + 1e-6)

        return volume, snr_db


def snr_positive_weight(
    snr_db: torch.Tensor,
    full_weight_db: float = 3.0,
    zero_weight_db: float = -6.0,
) -> torch.Tensor:
    """Weight of a positive pair, from the SNR of its worse view; linear ramp between the two.

    Deliberately unlike NRCL: a destroyed view is down-weighted, never turned into a negative.
    Pushing it away would teach the encoder that SNR itself is semantically meaningful.
    """
    if zero_weight_db >= full_weight_db:
        raise ValueError("zero_weight_db must be below full_weight_db.")
    return ((snr_db - zero_weight_db) / (full_weight_db - zero_weight_db)).clamp(0.0, 1.0)


class FixedNoiseTransform:
    """Noise-only injection at a fixed SNR, for measuring accuracy vs. SNR at test time."""

    def __init__(self, snr_db: Optional[float], correlated: bool = False, renormalize: bool = True):
        self.snr_db = snr_db
        self.correlated = correlated
        self.renormalize = renormalize

    def __call__(self, volume, return_info: bool = False):
        if not isinstance(volume, torch.Tensor):
            volume = torch.as_tensor(volume, dtype=torch.float32)
        out = volume.float()
        if self.snr_db is not None:
            out = add_correlated_noise(out, self.snr_db) if self.correlated else add_noise_at_snr(out, self.snr_db)
            if self.renormalize:
                out = (out - out.mean()) / (out.std() + 1e-6)
        if return_info:
            return out, ["noise"] if self.snr_db is not None else []
        return out

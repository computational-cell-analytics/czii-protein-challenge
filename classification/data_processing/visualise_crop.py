import json
import numpy as np
import mrcfile
from pathlib import Path
import napari
import time
import os
import zarr


def get_non_zarr(input_path):
    """Load .mrc tomogram from a folder containing exactly one .mrc file."""
    mrc_files = [f for f in os.listdir(input_path) if f.lower().endswith('.mrc')]
    
    if not mrc_files:
        raise FileNotFoundError(f"No .mrc file found in {input_path}")
    if len(mrc_files) > 1:
        raise ValueError(f"Multiple .mrc files found in {input_path}: {mrc_files}")
    
    mrc_path = os.path.join(input_path, mrc_files[0])
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        input_volume = mrc.data.astype(np.float32)
    return input_volume


def get_volume(input_path: str, zarr_: bool = False) -> np.ndarray:
    """Load a tomogram from either .zarr or .mrc format."""
    if zarr_:
        # Recursive search for .zarr folders
        zarr_folders = []
        for root, dirs, files in os.walk(input_path):
            for d in dirs:
                if d.endswith(".zarr"):
                    zarr_folders.append(os.path.join(root, d))
        
        if not zarr_folders:
            raise FileNotFoundError(f"No .zarr folder found under {input_path}")
        
        # Prefer denoised.zarr if it exists
        zarr_dir = next((f for f in zarr_folders if os.path.basename(f) == "denoised.zarr"), zarr_folders[0])
        
        # Append "0" subfolder
        zarr_path = os.path.join(zarr_dir, "0")
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Expected '0' subfolder inside {zarr_dir}, but not found.")
        
        zarr_file = zarr.open(zarr_path, mode="r")
        volume = zarr_file[:].astype(np.float32)
    else:
        volume = get_non_zarr(input_path)
    return volume


def load_no_class_coords(json_path):
    """Load coordinates from no_class.json."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    coords = []
    for p in data.get("points", []):
        loc = p.get("location", {})
        if all(k in loc for k in ["x", "y", "z"]):
            coords.append((loc["z"], loc["y"], loc["x"]))  # z,y,x order
    return np.array(coords)


def extract_crop(volume, center, crop_size=43):
    """Extract a 3D crop centered at 'center' from the tomogram."""
    half = crop_size // 2
    zc, yc, xc = map(int, np.round(center))
    zmin, zmax = zc - half, zc + half + 1
    ymin, ymax = yc - half, yc + half + 1
    xmin, xmax = xc - half, xc + half + 1

    # Handle boundaries with padding
    pad_width = (
        (max(0, -zmin), max(0, zmax - volume.shape[0])),
        (max(0, -ymin), max(0, ymax - volume.shape[1])),
        (max(0, -xmin), max(0, xmax - volume.shape[2])),
    )
    vol_padded = np.pad(volume, pad_width, mode='constant', constant_values=0)

    zmin_p, zmax_p = zmin + pad_width[0][0], zmax + pad_width[0][0]
    ymin_p, ymax_p = ymin + pad_width[1][0], ymax + pad_width[1][0]
    xmin_p, xmax_p = xmin + pad_width[2][0], xmax + pad_width[2][0]

    crop = vol_padded[zmin_p:zmax_p, ymin_p:ymax_p, xmin_p:xmax_p]
    return crop


def visualize_crops_sequential(tomogram_path, no_class_json, crop_size=43, n_crops=None, zarr_=False):
    """Open each crop in Napari sequentially, one after another."""
    volume = get_volume(tomogram_path, zarr_=zarr_)
    coords = load_no_class_coords(no_class_json)

    if len(coords) == 0:
        print("No coordinates found in no_class.json.")
        return

    if n_crops is not None:
        coords = coords[:n_crops]

    print(f"Loaded {len(coords)} coordinates from {no_class_json}")

    for i, coord in enumerate(coords, start=1):
        crop = extract_crop(volume, coord, crop_size)
        print(f"Showing crop {i}/{len(coords)} centered at {coord}")

        viewer = napari.Viewer(title=f"no_class crop {i}/{len(coords)}")
        viewer.add_image(
            crop,
            name=f"crop_{i}",
            contrast_limits=[np.min(crop), np.max(crop)],
            scale=(1, 1, 1),
        )
        viewer.add_points(
            np.array([[crop.shape[0] // 2, crop.shape[1] // 2, crop.shape[2] // 2]]),
            name="center",
            size=3,
            face_color='red'
        )

        napari.run()  # Waits until user closes the window
        time.sleep(0.2)  # small pause to avoid race conditions
        print(f"Closed viewer for crop {i}")

    print("All crops visualized.")


if __name__ == "__main__":
    
    tomogram_path = Path("/mnt/vast-nhr/home/muth9/u12095/test_crop/TS_69_2")
    no_class_json = Path("/mnt/vast-nhr/home/muth9/u12095/test_crop/no_class.json")

    # Number of crops to visualize (set None to show all)
    visualize_crops_sequential(tomogram_path, no_class_json, crop_size=43, n_crops=10, zarr_=True)

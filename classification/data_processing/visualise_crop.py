import json
import numpy as np
import mrcfile
from pathlib import Path
import napari
import time

def load_mrc_data(mrc_path):
    """Load a tomogram from an MRC file."""
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        data = mrc.data.astype(np.float32)
    return data

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

def visualize_crops_sequential(tomogram_path, no_class_json, crop_size=43, n_crops=None):
    """Open each crop in Napari sequentially, one after another."""
    volume = load_mrc_data(tomogram_path)
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
    # --- MODIFY THESE PATHS ---
    tomogram_path = Path("/mnt/vast-nhr/home/muth9/u12095/test_crop/tomo_rec_0_faket.mrc")
    no_class_json = Path("/mnt/vast-nhr/home/muth9/u12095/test_crop/no_class.json")

    # Number of crops to visualize (set None to show all)
    visualize_crops_sequential(tomogram_path, no_class_json, crop_size=43, n_crops=10)

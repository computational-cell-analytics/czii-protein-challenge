import os
import json
import random
import numpy as np
import mrcfile
from pathlib import Path
from scipy.spatial import cKDTree
import zarr
from scipy.ndimage import distance_transform_edt

def get_non_zarr(input_path):
    """
    Load a single volumetric file from a directory.
    Supports .mrc, .h5, .npy, .tif/.tiff.
    """
    supported_exts = ['.mrc', '.h5', '.npy', '.tif', '.tiff']
    valid_files = [f for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in supported_exts]
    print(f"valid_files {valid_files}")

    if not valid_files:
        raise FileNotFoundError(f"No supported files found in {input_path}. Supported extensions: {supported_exts}")

    file_path = os.path.join(input_path, valid_files[0])
    ext = os.path.splitext(file_path)[1].lower()
    print(f"file_path {file_path}")

    if ext == '.mrc':
        with mrcfile.open(file_path, permissive=True) as mrc:
            volume = mrc.data
    elif ext in ['.tif', '.tiff']:
        from tifffile import imread
        volume = imread(file_path)
    elif ext == '.npy':
        volume = np.load(file_path)
    elif ext == '.h5':
        from elf.io import open_file
        with open_file(file_path, "r") as f:
            keys = list(f.keys())
            if len(keys) == 1:
                key = keys[0]
            elif "data" in keys:
                key = "data"
            elif "raw" in keys:
                key = "raw"
            else:
                key = keys[0]
            volume = f[key][:]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return np.asarray(volume).astype(np.float32)


def get_volume(input_path: str, zarr_: bool = None) -> np.ndarray:
    """Load a tomogram, auto-detecting the format.

    If a ``.zarr`` folder is present under ``input_path`` it is used, otherwise
    the loader falls back to a single ``.mrc/.h5/.npy/.tif/.tiff`` file. The
    ``zarr_`` argument is kept for backwards compatibility but is ignored.
    """
    # Recursive search for .zarr folders
    zarr_folders = [
        os.path.join(root, d)
        for root, dirs, _ in os.walk(input_path)
        for d in dirs
        if d.endswith(".zarr")
    ]

    if zarr_folders:
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


def parse_json_files(json_files):
    """Parse multiple JSON files to extract coordinates."""
    coordinates = []
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            points = data.get("points", [])
            for point in points:
                loc = point.get("location", {})
                x, y, z = loc.get("x"), loc.get("y"), loc.get("z")
                if None not in (x, y, z):
                    coordinates.append((z / 10, y / 10, x / 10))
    return np.array(coordinates)


def generate_no_class_points(shape, existing_coords, min_distance=55, n_points=50):
    """
    Generate random points within tomogram volume that are:
      - ≥ min_distance from all existing coordinates and each other
      - ≥ min_distance from tomogram borders
    """
    z_max, y_max, x_max = shape
    existing_tree = cKDTree(existing_coords) if len(existing_coords) > 0 else None
    new_points = []
    tries = 0
    max_tries = 10000000

    while len(new_points) < n_points and tries < max_tries:
        tries += 1

        # Sample within edge-safe bounds
        candidate = np.array([
            random.uniform(min_distance, z_max - min_distance),
            random.uniform(min_distance, y_max - min_distance),
            random.uniform(min_distance, x_max - min_distance),
        ])

        # Check distance to all existing coordinates
        if existing_tree:
            dist, _ = existing_tree.query(candidate, k=1)
            if dist < min_distance:
                continue

        # Check distance to other new points
        if new_points:
            new_tree = cKDTree(new_points)
            dist, _ = new_tree.query(candidate, k=1)
            if dist < min_distance:
                continue

        new_points.append(candidate)

    if len(new_points) < n_points:
        print(f"Only generated {len(new_points)} points (target={n_points}) after {tries} attempts.")
    else:
        print(f"Generated {len(new_points)} points successfully.")

    return np.array(new_points)


def generate_no_class_points_from_distance_map(
    shape,
    existing_coords,
    min_distance=50,
    n_points=50,
    enforce_spacing=True
):
    """
    Generate points using a distance transform:
    - Points are sampled where distance to nearest protein ≥ min_distance
    - Optionally enforce spacing between sampled points
    """

    # Create empty volume
    volume = np.ones(shape, dtype=np.uint8)

    # Mark protein coordinates as 0
    coords_int = np.round(existing_coords).astype(int)

    # Clip to valid indices
    coords_int[:, 0] = np.clip(coords_int[:, 0], 0, shape[0]-1)
    coords_int[:, 1] = np.clip(coords_int[:, 1], 0, shape[1]-1)
    coords_int[:, 2] = np.clip(coords_int[:, 2], 0, shape[2]-1)

    volume[coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]] = 0

    # Compute distance transform
    distance_map = distance_transform_edt(volume).astype(np.float32)

    # Get valid candidate voxels 
    valid_mask = distance_map >= min_distance

    #remove edges #TODO maybe remove this? I do padding in training anyways, no?
    valid_mask[:min_distance, :, :] = False
    valid_mask[-min_distance:, :, :] = False
    valid_mask[:, :min_distance, :] = False
    valid_mask[:, -min_distance:, :] = False
    valid_mask[:, :, :min_distance] = False
    valid_mask[:, :, -min_distance:] = False
    candidates = np.argwhere(valid_mask)

    print(f"Found {len(candidates)} valid candidate voxels")

    if len(candidates) == 0:
        raise RuntimeError("No valid positions found with given min_distance")

    # Shuffle candidates
    np.random.shuffle(candidates)

    selected_points = []

    if not enforce_spacing:
        selected_points = candidates[:n_points]

    else:
        # Enforce spacing between new points
        tree = None

        for candidate in candidates:
            if len(selected_points) >= n_points:
                break

            candidate = candidate.astype(float)

            if tree is not None:
                dist, _ = tree.query(candidate, k=1)
                if dist < min_distance:
                    continue

            selected_points.append(candidate)

            if len(selected_points) > 1:
                tree = cKDTree(selected_points)

    print(f"Selected {len(selected_points)} points")

    return np.array(selected_points)


def create_no_class_json(input_dir, picks_dir, zarr_=False, n_points=50):
    """Create (or overwrite) no_class.json for a single tomogram folder."""
    # Load tomogram volume (supports mrc or zarr)
    volume = get_volume(input_dir, zarr_=zarr_)
    shape = volume.shape  # (z, y, x)

    # Collect all JSON files except no_class
    json_files = [str(p) for p in Path(picks_dir).glob("*.json") if not p.name.startswith("no_class") and not p.name.startswith("actin") and not p.name.startswith("mt")]
    existing_coords = parse_json_files(json_files)  # Angstrom to nm

    # Generate new points in nm
    no_class_points = generate_no_class_points_from_distance_map(
        shape,
        existing_coords,
        min_distance=40,
        n_points=n_points
    )

    # Build JSON data structure
    data = {
        "pickable_object_name": "no_class",
        "user_id": "synthetic",
        "session_id": "0",
        "run_name": Path(input_dir).name,
        "voxel_spacing": None,
        "unit": "angstrom",
        "points": []
    }

    #Convert nm to angstrom before saving
    for i, (z_nm, y_nm, x_nm) in enumerate(no_class_points):
        data["points"].append({
            "location": {
                "x": float(x_nm * 10),
                "y": float(y_nm * 10),
                "z": float(z_nm * 10),
            },
            "transformation_": np.eye(4).tolist(),
            "instance_id": i
        })

    # Always overwrite existing no_class.json
    output_path = Path(picks_dir) / "no_class.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Overwrote {output_path}")


def process_all_tomograms(tomo_root, gt_root, zarr_=False, n_points=50):
    """Iterate through all tomograms and regenerate no_class.json files."""
    for tomo_dir in Path(tomo_root).iterdir():
        if tomo_dir.is_dir():
            picks_dir = Path(gt_root) / tomo_dir.name #/ "Picks"
            if not picks_dir.exists():
                continue

            print(f"\n Processing {tomo_dir.name}")
            create_no_class_json(tomo_dir, picks_dir, zarr_=zarr_, n_points=n_points)


if __name__ == "__main__":
    tomo_root = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/ExperimentRuns_faket_dens1_5_distr_eqCl3"
    gt_root = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/ground_truth/structure_for_detection/ExperimentRuns_faket_dens1_5_distr_eqCl3"

    # Set zarr_=True if your data is in .zarr format
    process_all_tomograms(tomo_root, gt_root, zarr_=False, n_points=150)

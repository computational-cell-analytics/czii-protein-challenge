import os
import json
import random
import numpy as np
import mrcfile
from pathlib import Path
from scipy.spatial import cKDTree
import zarr

# =====================================================
# Volume loading utilities
# =====================================================

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

# =====================================================
# Coordinate generation logic
# =====================================================

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
    max_tries = 200000

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

def create_no_class_json(input_dir, picks_dir, zarr_=False, n_points=50):
    """Create (or overwrite) no_class.json for a single tomogram folder."""
    # Load tomogram volume (supports mrc or zarr)
    volume = get_volume(input_dir, zarr_=zarr_)
    shape = volume.shape  # (z, y, x)

    # Collect all JSON files except no_class
    json_files = [str(p) for p in Path(picks_dir).glob("*.json") if not p.name.startswith("no_class")]
    existing_coords = parse_json_files(json_files)  # Å → nm

    # Generate new points in nm
    no_class_points = generate_no_class_points(
        shape,
        existing_coords,
        min_distance=55,
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

    #Convert nm → Å before saving
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
            picks_dir = Path(gt_root) / tomo_dir.name / "Picks"
            if not picks_dir.exists():
                continue

            print(f"\n Processing {tomo_dir.name}")
            create_no_class_json(tomo_dir, picks_dir, zarr_=zarr_, n_points=n_points)

if __name__ == "__main__":
    tomo_root = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/ExperimentRuns"
    gt_root = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/public_test_dataset/ground_truth_scaled_for_detection/ExperimentRuns"

    # Set zarr_=True if your data is in .zarr format
    process_all_tomograms(tomo_root, gt_root, zarr_=True, n_points=25)

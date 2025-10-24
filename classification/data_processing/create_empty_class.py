import os
import json
import random
import numpy as np
import mrcfile
from pathlib import Path
from scipy.spatial import cKDTree

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
    Generate random points within shape that are:
      - at least min_distance away from existing coordinates and each other
      - at least min_distance away from tomogram edges
    """
    z_max, y_max, x_max = shape
    existing_tree = cKDTree(existing_coords) if len(existing_coords) > 0 else None
    new_points = []
    tries = 0
    max_tries = 200000

    while len(new_points) < n_points and tries < max_tries:
        tries += 1

        # Sample within valid bounds (avoid edges)
        candidate = np.array([
            random.uniform(min_distance, z_max - min_distance),
            random.uniform(min_distance, y_max - min_distance),
            random.uniform(min_distance, x_max - min_distance),
        ])

        # Check distance from existing protein coordinates
        if existing_tree:
            dist, _ = existing_tree.query(candidate, k=1)
            if dist < min_distance:
                continue

        # Check distance from already accepted no_class points
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

def create_no_class_json(tomogram_path, picks_dir, n_points=50):
    # Load tomogram to get shape
    with mrcfile.open(tomogram_path, permissive=True) as mrc:
        shape = mrc.data.shape  # (z, y, x)

    # Collect all other protein JSONs
    json_files = [str(p) for p in Path(picks_dir).glob("*.json") if not p.name.startswith("no_class")]
    existing_coords = parse_json_files(json_files)

    # Generate new valid no_class points
    no_class_points = generate_no_class_points(shape, existing_coords, min_distance=55, n_points=n_points)

    # Build JSON data structure
    data = {
        "pickable_object_name": "no_class",
        "user_id": "synthetic",
        "session_id": "0",
        "run_name": Path(tomogram_path).stem,
        "voxel_spacing": None,
        "unit": "angstrom",
        "points": []
    }

    for i, (z, y, x) in enumerate(no_class_points):
        data["points"].append({
            "location": {"x": float(x), "y": float(y), "z": float(z)},
            "transformation_": np.eye(4).tolist(),
            "instance_id": i
        })

    # Always overwrite existing no_class.json
    output_path = Path(picks_dir) / "no_class.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Overwrote {output_path}")

def process_all_tomograms(tomo_root, gt_root, n_points=50):
    """Iterate through all tomograms and regenerate no_class.json files."""
    for tomo_dir in Path(tomo_root).iterdir():
        if tomo_dir.is_dir():
            tomogram_files = list(tomo_dir.glob("*.mrc"))
            if not tomogram_files:
                continue

            tomogram_path = tomogram_files[0]
            picks_dir = Path(gt_root) / tomo_dir.name / "Picks"
            if not picks_dir.exists():
                continue

            print(f"\n Processing {tomo_dir.name}")
            create_no_class_json(tomogram_path, picks_dir, n_points)

if __name__ == "__main__":
    tomo_root = "/mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/ExperimentRuns_faket"
    gt_root = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/public_test_dataset/ground_truth_scaled_for_detection/ExperimentRuns_faket"
    process_all_tomograms(tomo_root, gt_root, n_points=25)

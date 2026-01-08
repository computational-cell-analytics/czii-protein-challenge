import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
import json
import argparse
import os
import zarr

def width_to_sigma(width, eps, lower_bound, upper_bound):
    # shrink needs to be between 0 and 1
    sigma = np.sqrt(-(width**2) / (2*np.log(eps)))
    #### bounding ####
    if lower_bound and upper_bound:
        if sigma < lower_bound:
            sigma = lower_bound
        elif sigma > upper_bound:
            sigma = upper_bound

    return int(sigma)

def create_gaussian_stamp(width, eps, lower_bound, upper_bound):
    """
    Creates a 3D Gaussian stamp (cube) with size width x width x width.
    If width is even, set width = width - 1.
    """
    if width % 2 == 0:
        width = width - 1
    
    sigma = width_to_sigma(width, eps, lower_bound, upper_bound)
    
    # Create a 3D matrix (stamp)
    stamp = np.zeros((width, width, width))
    center = width // 2
    
    # Set the center point to 1 (Gaussian peak)
    stamp[center, center, center] = 1

    # Apply 3D Gaussian filter to create the Gaussian distribution in 3D space
    #stamp = gaussian_filter(stamp, sigma=sigma)
    #Julia:
    stamp = gaussian(stamp, sigma=sigma, truncate=10.0, mode='constant')
    
    # Threshold the values based on epsilon and apply the scaling factor
    stamp[stamp < eps] = 0
    factor_3d = 2.5 #I had 1.6 as values for the centres of the gaussians before, but it should be 4, so 4/1.6=2.5
    stamp = stamp * 8 * factor_3d * np.pi * sigma**3

    return stamp
    
def parse_json_files(json_files):
    """
    Parse multiple JSON files to extract coordinates and protein names.

    Parameters:
        json_files (list of str): List of JSON file paths.

    Returns:
        list of tuples, list: Extracted coordinates and corresponding protein types.
    """
    coordinates = []
    protein_types = []

    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)

            points = data.get("points", [])
            for point in points:
                location = point.get("location", {})
                x = location.get("x")
                y = location.get("y")
                z = location.get("z")
                if x is not None and y is not None and z is not None:
                    coordinates.append((z / 10, y / 10, x / 10))  # Scale coordinates as needed
                    protein_types.append(data.get("pickable_object_name", "unknown"))

    return coordinates, protein_types

def create_width_dict():
    """
    Create a dictionary mapping protein names to widths.

    Returns:
        dict: Protein name to width mapping.
    """
    return {
        "apo-ferritin": 79.07,
        "beta-amylase": 33.27,
        "beta-galactosidase": 57.44,
        "ribosome": 109.02,
        "thyroglobulin": 76.47,
        "virus-like-particle": 79.07
    }

def precompute_gaussians(width_dict, eps, lower_bound, upper_bound):
    return {protein: create_gaussian_stamp(int(width * 0.3), eps, lower_bound, upper_bound) for protein, width in width_dict.items()}

def create_heatmap(json_folder, image_shape, eps=0.00001, sigma=None, lower_bound=None, upper_bound=None, bb=None):
    """
    Process a tomogram by creating a heatmap based on protein data from JSON files,
    with optional bounding box constraints.

    Parameters:
        json_folder (str): Path to the folder containing JSON files.
        image_shape (tuple): Shape of the 3D image (z, y, x).
        eps (float): Threshold for truncating the Gaussian.
        sigma (float, optional): Fixed sigma value for Gaussian.
        lower_bound (float, optional): Minimum allowed sigma.
        upper_bound (float, optional): Maximum allowed sigma.
        bb (tuple, optional): Bounding box (z_min, z_max, y_min, y_max, x_min, x_max).

    Returns:
        np.ndarray: Generated 3D heatmap.
    """
    picks_folder = os.path.join(json_folder, "Picks")
    json_files = [os.path.join(picks_folder, f) for f in os.listdir(picks_folder) if f.endswith('.json')]
    coordinates, protein_types = parse_json_files(json_files)
    width_dict = create_width_dict()
    gaussian_dict = precompute_gaussians(width_dict, eps, lower_bound, upper_bound)
    
    if bb:
        (z_min, z_max), (y_min, y_max), (x_min, x_max) = [(s.start, s.stop) for s in bb]
        restricted_shape = (z_max - z_min, y_max - y_min, x_max - x_min)
        heatmap = np.zeros(restricted_shape)
    else:
        heatmap = np.zeros(image_shape)
    
    for coord, protein in zip(coordinates, protein_types):
        z, y, x = map(int, coord)
        gaussian = gaussian_dict.get(protein, create_gaussian_stamp(1, eps, lower_bound, upper_bound))
        
        if bb and not (z_min <= z < z_max and y_min <= y < y_max and x_min <= x < x_max):
            continue
        
        z_offset, y_offset, x_offset = (z_min, y_min, x_min) if bb else (0, 0, 0)
        z, y, x = z - z_offset, y - y_offset, x - x_offset
        
        z_min_hm = max(0, z - gaussian.shape[0] // 2)
        z_max_hm = min(heatmap.shape[0], z + gaussian.shape[0] // 2 + 1)
        y_min_hm = max(0, y - gaussian.shape[1] // 2)
        y_max_hm = min(heatmap.shape[1], y + gaussian.shape[1] // 2 + 1)
        x_min_hm = max(0, x - gaussian.shape[2] // 2)
        x_max_hm = min(heatmap.shape[2], x + gaussian.shape[2] // 2 + 1)
        
        heatmap[z_min_hm:z_max_hm, y_min_hm:y_max_hm, x_min_hm:x_max_hm] = np.maximum(
            heatmap[z_min_hm:z_max_hm, y_min_hm:y_max_hm, x_min_hm:x_max_hm],
            gaussian[
                (z_min_hm - (z - gaussian.shape[0] // 2)):(z_max_hm - (z - gaussian.shape[0] // 2)),
                (y_min_hm - (y - gaussian.shape[1] // 2)):(y_max_hm - (y - gaussian.shape[1] // 2)),
                (x_min_hm - (x - gaussian.shape[2] // 2)):(x_max_hm - (x - gaussian.shape[2] // 2))
            ]
        )
    
    return heatmap

def get_label(json_folder, image_shape, eps=0.00001, sigma=None, lower_bound=None, upper_bound=None, bb=None):

    have_single_file = isinstance(json_folder, str)

    if have_single_file:
        return create_heatmap(json_folder, image_shape, eps, sigma, lower_bound, upper_bound, bb)
    else:
        print(f"len json folder {len(json_folder)}")
        return np.stack([create_heatmap(p, image_shape, eps, sigma, lower_bound, upper_bound, bb) for p in json_folder])


def get_tomo_shape(zarr_folder):
    """
    Get the shape of the tomogram stored under the '0' key in a Zarr folder.

    Parameters:
        zarr_folder (str): Path to the Zarr folder.

    Returns:
        tuple: Shape of the '0' tomogram array, or None if not found.
    """
    # Open the Zarr store
    zarr_store = zarr.open(zarr_folder, mode='r')

    # Iterate over groups and arrays in the Zarr store
    def traverse_zarr(group, parent_key=""):
        for key, item in group.items():
            current_key = f"{parent_key}/{key}" if parent_key else key
            if current_key != "0":
                continue

            if isinstance(item, zarr.core.Array):
                # Add array to Napari viewer
                data = np.array(item)
                return data.shape

            elif isinstance(item, zarr.hierarchy.Group):
                # Recursively traverse subgroup
                traverse_zarr(item, current_key)

    
    return traverse_zarr(zarr_store)

def main():

    parser = argparse.ArgumentParser(description="Generate a 3D heatmap from a tomogram and JSON files.")
    parser.add_argument("--zarr_path", "-z", type=str, required=True, help="Path to zarr file.")
    parser.add_argument("--json_folder", "-j", type=str, required=True, help="Path to the folder containing the 6 corresponding JSON files.")
    parser.add_argument("--output_folder", "-o", type=str, required=True, help="Path to save the generated heatmap as a .npy file.")

    args = parser.parse_args()

    tomogram_shape = get_tomo_shape(args.zarr_path)
    if tomogram_shape is not None:
        json_folder = args.json_folder
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)

        output_file =  os.path.join(output_folder, "heatmap.npy")

        heatmap = create_heatmap(json_folder, tomogram_shape)

        # Save the heatmap to a file
        np.save(output_file, heatmap)
        print(f"Heatmap saved to {output_file}")

        '''import napari
        loaded_array = np.load(output_file)
        v = napari.Viewer()
        v.add_image(loaded_array, name= f"heatmap")
        napari.run()'''

    else:
        print("No 0 zarr was found")

if __name__ == "__main__":
    main()
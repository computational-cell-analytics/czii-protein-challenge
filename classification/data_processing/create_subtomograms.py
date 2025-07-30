import argparse
import json
import os

import numpy as np
import napari
import zarr


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Extract subtomograms from raw zarr data using peak coordinates.')
    parser.add_argument('--raw', type=str, required=True, help='Path to the raw Zarr dataset directory.')
    parser.add_argument('--heatmap', type=str, required=True, help='Path to the .npy heatmap file (3D).')
    parser.add_argument('--peaks', type=str, required=True, help='Path to the .json file with 3D peak coordinates.')
    return parser.parse_args()


def get_volume(input_path):
    """
    Load 3D volume from Zarr file structure.
    Assumes structure: input_path/VoxelSpacing10.000/denoised.zarr/0/
    """
    zarr_path = os.path.join(input_path, "VoxelSpacing10.000", "denoised.zarr", "0")
    zarr_file = zarr.open(zarr_path, mode='r')
    return zarr_file[:]


def load_heatmap(npy_filepath):
    """Load heatmap from .npy file and squeeze singleton dimensions."""
    heatmap = np.load(npy_filepath)
    print("Original heatmap shape:", heatmap.shape)
    heatmap = np.squeeze(heatmap)
    print("Squeezed heatmap shape:", heatmap.shape)
    return heatmap



def load_peaks(json_filepath):
    """Load list of 3D peak coordinates from JSON file."""
    with open(json_filepath, 'r') as f:
        peaks = json.load(f)
    return [tuple(map(int, peak)) for peak in peaks]

def estimate_gaussian_extent(heatmap, coord, threshold=0.1): #TODO might want to lower threshold?
    """
    Estimate the size of the Gaussian blob in each dimension based on intensity threshold.

    Returns:
        bbox_size (int): Max extent across axes with buffer.
    """
    z, y, x = coord
    peak_value = heatmap[z, y, x]
    extent = []

    for axis, dim in enumerate(heatmap.shape):
        size = 0
        for direction in [-1, 1]:
            offset = 1
            while True:
                idx = [z, y, x]
                idx[axis] += direction * offset
                if 0 <= idx[axis] < heatmap.shape[axis]:
                    val = heatmap[tuple(idx)]
                    if val >= threshold * peak_value:
                        size += 1
                        offset += 1
                    else:
                        break
                else:
                    break
        extent.append(size)

    return max(extent)


def extract_subtomograms(raw_data, peaks, size, halo=4):
    """Extract centered subtomograms from raw data."""
    bbox_size = size + halo
    print(f"Using bounding box size (with halo): {bbox_size}")

    subtomograms = []
    offset = size // 2
    extra = 1 if size % 2 != 0 else 0
    half_size = offset + halo

    for z, y, x in peaks:
        zmin = max(0, z - half_size)
        zmax = min(raw_data.shape[0], z + half_size + extra)
        ymin = max(0, y - half_size)
        ymax = min(raw_data.shape[1], y + half_size + extra)
        xmin = max(0, x - half_size)
        xmax = min(raw_data.shape[2], x + half_size + extra)
        
        cube = raw_data[zmin:zmax, ymin:ymax, xmin:xmax]
        subtomograms.append(cube)

    return subtomograms


def visualize_with_napari(raw_data, heatmap, peaks, subtomograms):
    """Launch Napari viewer with raw data, heatmap, peaks, and extracted subtomograms."""
    #viewer = napari.Viewer()
    #viewer.add_image(raw_data, name='Raw Volume', colormap='gray')
    #viewer.add_image(heatmap, name='Heatmap', colormap='magma')
    #viewer.add_points(peaks, name='Peaks', size=5, face_color='cyan')

    for i, tomo in enumerate(subtomograms):
        viewer = napari.Viewer()
        viewer.add_image(tomo, name=f'Subtomo {i}')
        napari.run()

    #napari.run()

def get_max_extent(peaks, heatmap):
    """get max dimensions for the bb that includes all proteins. Makes sure the max extent is not an outlier. """

    max_extent = 0
    all_extents = []

    for coord in peaks:
        extent = estimate_gaussian_extent(heatmap, coord)
        all_extents.append(extent)

    # Convert to array for robust statistics
    extents_arr = np.array(all_extents)
    median = np.median(extents_arr)
    std = np.std(extents_arr)

    # Filter out outliers: within 1.5 standard deviations of the median
    filtered_extents = extents_arr[np.abs(extents_arr - median) <= 1.5 * std]

    # Fallback if filtering removes all
    if len(filtered_extents) == 0:
        max_extent = int(np.max(extents_arr))
    else:
        max_extent = int(np.max(filtered_extents))
    
    return max_extent

def main():
    args = parse_arguments()

    raw_data = get_volume(args.raw)
    heatmap = load_heatmap(args.heatmap)
    peaks = load_peaks(args.peaks)

    print(f"Loaded {len(peaks)} peaks.")

    max_extent = get_max_extent(peaks, heatmap)

    subtomograms = extract_subtomograms(raw_data, peaks, max_extent)
    visualize_with_napari(raw_data, heatmap, peaks, subtomograms)


if __name__ == '__main__':
    main()

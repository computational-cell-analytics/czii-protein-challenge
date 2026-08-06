from setuptools import setup, find_packages

setup(
    name="pro-revelio",
    version="0.1.0",
    author="Sarah Muth",
    url="https://github.com/computational-cell-analytics/czii-protein-challenge",
    packages=find_packages(),
    package_data={"pro_revelio_napari": ["napari.yaml"]},
    entry_points={"napari.manifest": ["pro-revelio = pro_revelio_napari:napari.yaml"]},
)

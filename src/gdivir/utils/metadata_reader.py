"""
This module provides utilities for reading and handling metadata related to datasets.

It includes functions and classes to read metadata files, manage datasets, and handle
directory structures for storing data.

Classes
-------
Dataset
    Represents a dataset with methods to access metadata and years of data.
Metadata
    Manages multiple datasets and provides access to their metadata.
Directories
    Manages directory paths for storing different types of data.

Functions
---------
read_metadata_file(file_name: str) -> dict
    Reads a YAML metadata file and returns its content as a dictionary.
find_metadata_version(versions: dict, year: int) -> dict
    Finds the metadata version for a given year.
"""

from typing import Literal
from pathlib import Path

import yaml


__all__ = [
    "_Dataset",
    "_Metadata",
    "_RegionType",
    "_ExternalDataset",
    "metadata",
    "directories",
]

PAKAGE_PATH = Path(__file__).parents[1]

_Dataset = Literal["geographical_divisions", "census_results"]
_Metadata = Literal["raw_files", "tables", "external_datasets"]
_RegionType = Literal["Province", "County", "District", "Rural_District"]
_ExternalDataset = Literal[
    "hbsir",
]

with (PAKAGE_PATH / "config/settings.yaml").open(encoding="utf-8") as yaml_file:
    config: dict = yaml.safe_load(yaml_file)


def read_metadata_file(file_name: str) -> dict:
    """Reads a YAML metadata file and returns its content as a dictionary.

    Parameters
    ----------
    file_name : str
        The name of the metadata file (without extension) to read.

    Returns
    -------
    dict
        The content of the metadata file.
    """
    with (PAKAGE_PATH / f"metadata/{file_name}.yaml").open(encoding="utf-8") as yaml_file:
        file_content = yaml.safe_load(yaml_file)
    return file_content


def find_metadata_version(versions: dict, year: int) -> dict:
    """Finds the metadata version for a given year.

    Parameters
    ----------
    versions : dict
        A dictionary where keys are years and values are metadata versions.
    year : int
        The reference year.

    Returns
    -------
    dict
        The metadata version for the given year.

    Raises
    ------
    AssertionError
        If no valid metadata version is found for the given year.
    """
    last_version = 0
    for version_year in versions.keys():
        if (version_year <= year) & (last_version <= year):
            last_version = version_year
    assert last_version != 0
    return versions[last_version]


class Dataset:
    def __init__(self, name: str, raw_files: dict, tables: dict) -> None:
        """Initializes a Dataset instance.

        Parameters
        ----------
        name : str
            The name of the dataset.
        raw_files : dict
            A dictionary containing raw file metadata.
        tables : dict
            A dictionary containing table metadata.
        """
        self.name = name
        self.raw_files: dict = raw_files[name]
        self.tables: dict = tables[name]

    @property
    def years(self) -> list[int]:
        """Returns a list of years for which the dataset has data.

        Returns
        -------
        list[int]
            A list of years.
        """
        return list(self.raw_files.keys())
    
    def get_next_year(self, year: int) -> int:
        """Returns the next available year after the given year.

        Parameters
        ----------
        year : int
            The reference year.

        Returns
        -------
        int
            The next available year.
        """
        next_year_dict = dict(zip(self.years[:-1], self.years[1:]))
        return next_year_dict[year]
    
    def get_previous_year(self, year: int) -> int:
        """Returns the previous available year before the given year.

        Parameters
        ----------
        year : int
            The reference year.

        Returns
        -------
        int
            The previous available year.
        """
        next_year_dict = dict(zip(self.years[1:], self.years[:-1]))
        return next_year_dict[year]
    
    def get_nearest_year(self, year: int, prefer_later: bool = True) -> int:
        """Returns the nearest available year to the given year.

        Parameters
        ----------
        year : int
            The reference year.
        prefer_later : bool, optional
            If True, prefer a later year if there is a tie. Default is True.

        Returns
        -------
        int
            The nearest available year.
        """
        distances = [abs(dataset_year - year) for dataset_year in self.years]
        min_distance = min(distances)
        indices = [
            index for index, distance
            in enumerate(distances)
            if distance == min_distance
        ]
        near_years = [self.years[index] for index in indices]
        if prefer_later:
            nearest_year = max(near_years)
        else:
            nearest_year = min(near_years)
        return nearest_year

    def get_metadata_version(self, metadata: _Metadata, year: int) -> dict:
        """Returns the metadata version for a given year.

        Parameters
        ----------
        metadata : _Metadata
            The type of metadata (raw_files or tables).
        year : int
            The reference year.

        Returns
        -------
        dict
            The metadata version for the given year.
        """
        if metadata == "raw_files":
            return find_metadata_version(self.raw_files, year)
        if metadata == "tables":
            return find_metadata_version(self.tables, year)
        raise ValueError


class Metadata: 
    raw_files: dict = read_metadata_file("raw_files")
    tables: dict = read_metadata_file("tables")
    external_datasets: dict = read_metadata_file("external_datasets")

    def __init__(self) -> None:
        """Initializes a Metadata instance."""
        self.geographical_divisions = Dataset(
            "geographical_divisions",
            self.raw_files,
            self.tables,
        )
        self.census_results = Dataset(
            "census_results",
            self.raw_files,
            self.tables,
        )

    def __getitem__(self, dataset: _Dataset) -> Dataset:
        """Returns the Dataset instance for the given dataset name.

        Parameters
        ----------
        dataset : _Dataset
            The name of the dataset.

        Returns
        -------
        Dataset
            The Dataset instance.
        """
        if dataset == "geographical_divisions":
            return self.geographical_divisions
        if dataset == "census_results":
            return self.census_results
        raise ValueError


class Directories:
    root = Path(config["data_path"])
    original_data: Path
    raw_data: Path
    cleaned_data: Path
    geographical_divisions: Path
    census_results: Path

    internal_data = Path(__file__).parents[1].joinpath("internal_data")
    internal_results = internal_data.joinpath("results")

    def __init__(self) -> None:
        """Initializes a Directories instance and creates necessary directories."""
        self.original_data = self.root.joinpath("1_original")
        self.raw_data = self.root.joinpath("2_raw")
        self.cleaned_data = self.root.joinpath("3_cleaned")
        self.geographical_divisions = self.cleaned_data.joinpath("geographical_divisions.parquet")
        self.census_results = self.cleaned_data.joinpath("census_results.parquet")

        self.root.mkdir(parents=True, exist_ok=True)
        self.original_data.mkdir(exist_ok=True)
        self.raw_data.mkdir(exist_ok=True)
        self.cleaned_data.mkdir(exist_ok=True)


metadata = Metadata()
directories = Directories()

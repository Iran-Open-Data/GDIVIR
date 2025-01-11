from typing import Literal
from pathlib import Path

import yaml

from .metadata_utils import find_metadata_version


_Dataset = Literal["geographical_divisions", "census_results"]
_Metadata = Literal["raw_files", "tables", "external_datasets"]
_RegionType = Literal["Province", "County", "District", "Rural_District"]
_ExternalDataset = Literal[
    "hbsir",
]

with Path("gdivir", "config", f"settings.yaml").open(encoding="utf-8") as yaml_file:
    config: dict = yaml.safe_load(yaml_file)


def read_metadata_file(file_name: str) -> dict:
    with Path("gdivir", "metadata", f"{file_name}.yaml").open(encoding="utf-8") as yaml_file:
        file_content = yaml.safe_load(yaml_file)
    return file_content


class Dataset:
    def __init__(self, name: str, raw_files: dict, tables: dict) -> None:
        self.name = name
        self.raw_files: dict = raw_files[name]
        self.tables: dict = tables[name]

    @property
    def years(self) -> list[int]:
        return list(self.raw_files.keys())
    
    def get_next_year(self, year: int) -> int:
        next_year_dict = dict(zip(self.years[:-1], self.years[1:]))
        return next_year_dict[year]
    
    def get_previous_year(self, year: int) -> int:
        next_year_dict = dict(zip(self.years[1:], self.years[:-1]))
        return next_year_dict[year]
    
    def get_nearest_year(self, year: int, prefer_later: bool = True) -> int:
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

    def __init__(self) -> None:
        self.original_data = self.root.joinpath("original")
        self.raw_data = self.root.joinpath("raw")
        self.cleaned_data = self.root.joinpath("cleaned")
        self.geographical_divisions = self.cleaned_data.joinpath("geographical_divisions.parquet")
        self.census_results = self.cleaned_data.joinpath("census_results.parquet")

        self.root.mkdir(parents=True, exist_ok=True)
        self.original_data.mkdir(exist_ok=True)
        self.raw_data.mkdir(exist_ok=True)
        self.cleaned_data.mkdir(exist_ok=True)


metadata = Metadata()
directories = Directories()

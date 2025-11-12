from pathlib import Path

import pandas as pd

from ..utils import _ExternalDataset, directories


def create_geodiv_mapping(external_dataset: _ExternalDataset) -> dict[int, dict]:
    geodiv_standard = find_geodiv_standard(external_dataset)
    geodiv_provinces = (
        pd.read_parquet(directories.geographical_divisions)
        .loc[lambda df: df["Region_Type"].eq("Province")]
    )
    file_path = Path(__file__).parents[1].joinpath(
        "internal_data",
        "datasets",
        external_dataset,
        "provinces.csv",
    )
    (
        pd.read_csv(file_path, dtype={"Province_Code": "str"})
        .set_index("County_Code")
        .rename(lambda name: int(name), axis="columns")
    )


def find_geodiv_standard(external_dataset: _ExternalDataset) -> dict:
    geodiv_provinces = (
        pd.read_parquet(directories.geographical_divisions)
        .loc[lambda df: df["Region_Type"].eq("Province")]
        .pivot(index="Year", columns="ID", values="Province_ID")
        .notna()
        .drop_duplicates()
        .apply(lambda s: set(s.loc[s].index), axis="columns")
    )
    file_path = Path(__file__).parents[1].joinpath(
        "internal_data",
        "datasets",
        external_dataset,
        "provinces.csv",
    )
    return (
        pd.read_csv(file_path, dtype={"ID": "str"})
        .set_index("ID")
        .rename(lambda name: int(name), axis="columns")
        .notna()
        .transpose()
        .apply(lambda s: set(s.loc[s].index), axis="columns")
        .apply(lambda s: geodiv_provinces.eq(s).loc[lambda r: r].index[0])
        .to_dict()
    )

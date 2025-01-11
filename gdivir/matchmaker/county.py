from pathlib import Path

import pandas as pd

from gdivir.matchmaker import common, province
from ..utils import _ExternalDataset, directories, metadata


def find_geodiv_standard(external_dataset: _ExternalDataset) -> dict:
    provincial_standards = province.find_geodiv_standard("hbsir")

    geodiv_counties: pd.DataFrame = (
        pd.read_parquet(directories.geographical_divisions)
        .astype({"Year": int})
        .loc[lambda df: df["Region_Type"].eq("County")]
        .pivot(index="Year", columns="ID", values="County_ID")
        .notna()
        .drop_duplicates()
        .apply(lambda s: set(s.loc[s].index), axis="columns")
        .to_frame("Counties")
    )
    ept_mappings = build_annual_ept_mappings(provincial_standards)

    file_path = Path(__file__).parents[1].joinpath(
        "internal_data",
        "datasets",
        external_dataset,
        "counties.csv",
    )
    return (
        pd.read_csv(file_path, dtype={"ID": "str"})
        .set_index("ID")
        .rename(lambda name: int(name), axis="columns")
        .notna()
        .transpose()
        .apply(get_counties_code, external_dataset=external_dataset, axis="columns")
        .apply(
            _count_differences,
            geodiv_counties=geodiv_counties,
            ept_mappings=ept_mappings,
        )
        .apply(lambda s: s.loc[s.eq(0)].index[0], axis="columns")
        .to_dict()
    )


def _replace_values_in_set(input_set: set, mapping_dict: int) -> set:
    return {mapping_dict.get(item, item) for item in input_set}


def get_counties_code(s: pd.Series, external_dataset) -> set:
    non_standard_codes = (
        metadata.external_datasets
        .get(external_dataset, {})
        .get("counties", {})
        .get("non_standard_codes", {})
    )
    return _replace_values_in_set(
        set(s.loc[s].index),
        non_standard_codes.get(s.name, {}),
    )


def _count_differences(
    dataset_counties: set,
    geodiv_counties: dict,
    ept_mappings: dict,
) -> dict:
    return (
        geodiv_counties
        .loc[lambda df: df.index >= 1363]
        .apply(
            lambda s:
            _replace_values_in_set(dataset_counties, ept_mappings[s.name])
            .difference(s["Counties"])
            ,
            axis="columns"
        )
        .apply(lambda s: len(s))
    )


def build_annual_ept_mappings(provincial_standards: dict) -> dict:
    epts = _get_extra_provincial_transformations(provincial_standards)
    annual_ept_mappings = {}
    for year in provincial_standards:
        annual_ept_mappings[year] = {}
        for standard_year, mapping in epts.items():
            annual_ept_mappings[year].update(mapping) if year < standard_year else None
    return annual_ept_mappings


def _get_extra_provincial_transformations(provincial_standards: dict) -> dict:
    standard_years = set(provincial_standards.values()).difference({1365})
    transformations = {1365: {}}
    for year in standard_years:
        transformations[year] = (
            common.create_one_to_one_mapping_documentation(year, "County")
            .loc[lambda df: df["Selected"]]
            .loc[lambda df: df["New_County_ID"].str[:2] != df["Old_County_ID"].str[:2]]
            .set_index("New_County_ID")
            .loc[:, "Old_County_ID"]
            .to_dict()
        )
    return transformations


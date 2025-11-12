from pathlib import Path

import pandas as pd

from .. import matchmaker
from ..utils import directories, metadata


def export_hbsir_standard_province_mapping() -> None:
    mapping_table = create_province_mapping_table()
    mapping_dict = create_mapping_dict(mapping_table)
    mapping_string = create_mapping_string(mapping_dict)

    file_path = Path("results", "hbsir_standard_province_mapping.yaml")
    file_path.parent.mkdir(exist_ok=True)
    with file_path.open(mode="w") as file:
        file.write(mapping_string)


def export_hbsir_standard_county_mapping() -> None:
    mapping_table = create_county_mapping_table()
    mapping_dict = create_mapping_dict(mapping_table)
    mapping_string = create_mapping_string(mapping_dict)

    file_path = Path("results", "hbsir_standard_county_mapping.yaml")
    file_path.parent.mkdir(exist_ok=True)
    with file_path.open(mode="w") as file:
        file.write(mapping_string)


def create_province_mapping_table() -> pd.DataFrame:
    file_path = directories.internal_data.joinpath("datasets", "hbsir", "provinces.csv")
    province_geodiv_standard = matchmaker.province.find_geodiv_standard("hbsir")
    geodiv_provinces = (
        pd.read_parquet(directories.geographical_divisions)
        .astype({"Year": int})
        .loc[lambda df: df["Region_Type"].eq("Province")]
        .loc[:, ["Year", "ID"]]
        .rename(columns={
            "Year": "GeoDiv_Year",
            "ID": "GeoDiv_ID",
        })
    )
    return (
        pd.read_csv(file_path, dtype={"ID": "str"})
        .set_index("ID")
        .rename(lambda name: int(name), axis="columns")
        .rename_axis("Year", axis="columns")
        .stack()
        .reorder_levels(["Year", "ID"])
        .sort_index()
        .index.to_frame()
        .reset_index(drop=True)
        .assign(GeoDiv_Year = lambda df: df["Year"].replace(province_geodiv_standard))
        .rename(columns={
            "Year": "Dataset_Year",
            "ID": "Dataset_ID",
        })
        .merge(
            geodiv_provinces,
            how="left",
            left_on=["GeoDiv_Year", "Dataset_ID"],
            right_on=["GeoDiv_Year", "GeoDiv_ID"],
        )
    )


def create_county_mapping_table(external_dataset: str = "hbsir") -> pd.DataFrame:
    province_geodiv_standard = matchmaker.province.find_geodiv_standard("hbsir")
    county_geodiv_standard = matchmaker.county.find_geodiv_standard("hbsir")
    geodiv_counties = (
        pd.read_parquet(directories.geographical_divisions)
        .astype({"Year": int})
        .loc[lambda df: df["Region_Type"].eq("County")]
        .loc[:, ["Year", "ID"]]
        .rename(columns={
            "Year": "GeoDiv_Year",
            "ID": "GeoDiv_ID",
        })
    )
    file_path = r"gdivir\internal_data\datasets\hbsir\counties.csv"
    non_standard_codes = (
        metadata.external_datasets
        .get(external_dataset, {})
        .get("counties", {})
        .get("non_standard_codes", {})
    )
    ept_mappings = matchmaker.county.build_annual_ept_mappings(province_geodiv_standard)

    return (
        pd.read_csv(file_path, dtype={"ID": "str"})
        .set_index("ID")
        .rename(lambda name: int(name), axis="columns")
        .rename_axis("Year", axis="columns")
        .stack()
        .reorder_levels(["Year", "ID"])
        .sort_index()
        .index.to_frame()
        .reset_index(drop=True)
        .assign(GeoDiv_Year = lambda df: df["Year"].replace(county_geodiv_standard))
        .rename(columns={
            "Year": "Dataset_Year",
            "ID": "Dataset_ID",
        })
        .assign(
            Corrected_ID = lambda df: df
            .apply(
                lambda s:
                non_standard_codes.get(s["Dataset_Year"], {})
                .get(s["Dataset_ID"], s["Dataset_ID"])
                ,
                axis="columns",
            )
        )
        .assign(
            Corrected_ID = lambda df: df
            .apply(
                lambda s:
                ept_mappings.get(s["GeoDiv_Year"], {})
                .get(s["Corrected_ID"], s["Corrected_ID"])
                ,
                axis="columns",
            )
        )
        .merge(
            geodiv_counties,
            how="left",
            left_on=["GeoDiv_Year", "Corrected_ID"],
            right_on=["GeoDiv_Year", "GeoDiv_ID"],
        )
    )


def create_mapping_dict(mapping_table: pd.DataFrame) -> dict:
    mapping_series = (
        mapping_table
        .drop_duplicates(["Dataset_ID", "GeoDiv_ID"])
        .assign(Count=lambda df: df.groupby("Dataset_ID")["Dataset_Year"].transform("count"))
        .apply(
            lambda s:
            {s["Dataset_ID"]: s["GeoDiv_ID"]} if s["Count"] == 1 else
            {s["Dataset_ID"]: {s["Dataset_Year"]: s["GeoDiv_ID"]}}
            ,
            axis="columns"
        )
    )

    mapping = {}
    for mapping_part in mapping_series:
        for code in mapping_part:
            if code in mapping:
                mapping[code].update(mapping_part[code])
            else:
                mapping.update(mapping_part)
    return mapping


def create_mapping_string(mapping: dict) -> str:
    mapping_string = ""
    mapping = dict(sorted(mapping.items()))
    for dataset_code, translation in mapping.items():
        mapping_string += f"{int(dataset_code)}:"
        if isinstance(translation, str):
            mapping_string += f" '{translation}'\n"
        elif isinstance(translation, dict):
            mapping_string += "\n"
            for year, code in translation.items():
                mapping_string += f"  {year}: '{code}'\n"
    return mapping_string

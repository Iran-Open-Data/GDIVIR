from typing import Iterable

import pandas as pd

from ..data_handler import load_dataset
from ..data_cleaner import normalize_text
from ..utils.metadata_reader import directories


def create_province_version_table() -> None:
    province_pivot = (
        load_dataset()
        .loc[lambda df: df["Region_Type"].eq("Province")]
        .pivot(columns="ID", index="Year", values="Province_Name")
    )
    normalized_pivot = province_pivot.apply(normalize_text)
    province_pivot_index = normalized_pivot.drop_duplicates(keep="last").index
    province_pivot_years = normalized_pivot.drop_duplicates(keep="first").index
    (
        province_pivot
        .reindex(province_pivot_index)
        .set_axis(province_pivot_years)
        .transpose()
        .to_csv(directories.internal_results / "provinces_versions.csv")
    )


def create_counties_version_table() -> None:
    for i in range(31):
        _create_counties_version_table_for_province(f"{i:0>2}")


def _create_counties_version_table_for_province(province_code: str) -> None:
    pivot = (
        load_dataset()
        .loc[lambda df: df["Region_Type"].eq("County")]
        .loc[lambda df: df["Province_ID"].eq(province_code)]
        .pivot(columns="ID", index="Year", values="County_Name")
    )
    normalized_pivot = pivot.apply(normalize_text)
    pivot_index = normalized_pivot.drop_duplicates(keep="last").index
    pivot_years = normalized_pivot.drop_duplicates(keep="first").index
    (
        pivot
        .reindex(pivot_index)
        .set_axis(pivot_years)
        .transpose()
        .to_csv(directories.internal_results / f"{province_code}_counties_versions.csv")
    )


def get_province_version_table() -> pd.DataFrame:
    return (
        pd.read_csv(
            directories.internal_results / "provinces_versions.csv",
            dtype=str
        )
        .set_index("ID")
    )


def search_province_version_year(items: Iterable[str]) -> str:
    province_version_table = get_province_version_table()
    province_version_count = province_version_table.count()
    items_list = list(items)
    items_len = len(items_list)
    if items_len in province_version_count.values:
        province_versions = province_version_count.loc[lambda s: s.eq(items_len)].index.tolist()
    else:
        raise KeyError
    if not len(province_versions) == 1:
        raise KeyError
    province_version = province_versions[0]
    return province_version


def extract_province_codes(items: Iterable[str]) -> list[str]:
    version_table = get_province_version_table()
    version_year = search_province_version_year(items)
    code_mapping = (
        version_table[version_year]
        .pipe(normalize_text)
        .reset_index()
        .set_index(version_year)
        .loc[:, "ID"]
        .to_dict()
    )
    codes = (
        pd.Series(list(items), dtype=str)
        .pipe(normalize_text)
        .map(code_mapping)
    )
    assert codes.isna().sum() == 0
    codes_list = codes.to_list()
    return codes_list


def get_county_version_table(province_code: str) -> pd.DataFrame:
    return (
        pd.read_csv(
            directories.internal_results / f"{province_code}_counties_versions.csv",
            dtype=str
        )
        .set_index("ID")
    )


def search_county_version_year(items: Iterable[str], province_code: str) -> str:
    version_table = get_county_version_table(province_code)
    version_count = version_table.count()
    items_list = list(items)
    items_len = len(items_list)
    if items_len in version_count.values:
        versions = version_count.loc[lambda s: s.eq(items_len)].index.tolist()
    else:
        raise KeyError
    if not len(versions) == 1:
        raise Exception
    version = versions[0]
    return version


def extract_county_codes(items: Iterable[str], province_code: str) -> list[str]:
    version_table = get_county_version_table(province_code)
    version_year = search_county_version_year(items, province_code)
    code_mapping = (
        version_table[version_year]
        .pipe(normalize_text)
        .reset_index()
        .set_index(version_year)
        .loc[:, "ID"]
        .to_dict()
    )
    codes = (
        pd.Series(list(items), dtype=str)
        .pipe(normalize_text)
        .map(code_mapping)
    )
    assert codes.isna().sum() == 0
    codes_list = codes.to_list()
    return codes_list

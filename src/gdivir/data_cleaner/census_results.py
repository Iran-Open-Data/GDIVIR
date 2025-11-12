import pandas as pd

from ..utils import directories
from . import common

CLEAN_TABLE_COLUMNS = [
    "Year",
    "ID",
    "Province_ID",
    "Province_Name",
    "County_ID",
    "County_Name",
    "District_ID",
    "District_Name",
    "Rural_District_or_City_ID",
    "Rural_District_or_City_Name",
    "Village_ID",
    "Village_Name",
    "Region_Type",
    "Household_Count",
    "Population",
]


def create_clean_table(year: int) -> pd.DataFrame:
    path = directories.raw_data / "census_results" / f"{year}.csv"
    table = pd.read_csv(path, dtype=str)

    table = table.fillna("")
    common.extract_ids_from_long_id(table, dataset="census_results", year=year)
    if (year >= 1365) and (year <= 1390):
        table["Village_ID"] = table["Village_ID"].str.slice(3)
    common.create_long_id(table)
    common.create_rural_district_or_city_name(table)
    common.create_region_type_column(table)
    common.set_region_type_labels(table)
    table = table.replace("", None)
    table["Household_Count"] = (
        table["Household_Count"]
        .replace("\\D", "", regex=True)
        .replace("", None)
        .astype("UInt64")
    )
    table["Population"] = (
        table["Population"]
        .replace("\\D", "", regex=True)
        .replace("", None)
        .astype("UInt64")
    )
    if year <= 1390:
        table = pd.concat(
            [
                table,
                create_city_records_with_districts(table, year),
            ]
        )

    table["Year"] = year
    table = table.loc[:, CLEAN_TABLE_COLUMNS]
    assert isinstance(table, pd.DataFrame)
    table = table.sort_values("ID")
    return table


def create_city_records_with_districts(table: pd.DataFrame, year: int) -> pd.DataFrame:
    return (
        table
        .loc[lambda df: df["Region_Type"].eq("City_District")]
        .assign(
            Rural_District_or_City_Name=lambda df:
            df["Rural_District_or_City_Name"]
            .str.replace("\\d", "", regex=True)
            .str.strip()
            ,
            City_Name=lambda df:
            df["Rural_District_or_City_Name"]
            .str.replace("آ", "ا")
            .str.replace("\\(.+\\)", "", regex=True)
            .str.replace(" ", "")
            ,
            ID=lambda df: df["ID"].str[:-4]
        )
        .assign(Province_County=lambda df: df["ID"].str[:4])
        .groupby(["City_Name", "Province_County"])
        .aggregate(
            {
                "ID": "first",
                "Province_ID": "first",
                "Province_Name": "first",
                "County_ID": "first",
                "County_Name": "first",
                "District_ID": "first",
                "District_Name": "first",
                "Village_ID": "first",
                "Village_Name": "first",
                "Household_Count": "sum",
                "Population": "sum",
            }
        )
        .join(get_city_info_from_geodiv(year))
        .assign(ID=lambda df: df["ID"] + df["Rural_District_or_City_ID"])
        .reset_index(drop=True)
    )


def get_city_info_from_geodiv(year: int) -> pd.DataFrame:
    return (
        pd.read_parquet(
            directories.geographical_divisions,
            filters=[("Year", "=", year)],
        )
        .loc[lambda df: df["Region_Type"].eq("City")]
        .assign(
            City_Name=lambda df:
            df["Rural_District_or_City_Name"]
            .str.replace("آ", "ا")
            .str.replace(" ", "")
        )
        .assign(Province_County=lambda df: df["ID"].str[:4])
        .set_index(["City_Name", "Province_County"])
        .loc[
            :,
            [
                "Rural_District_or_City_Name",
                "Rural_District_or_City_ID",
                "Region_Type",
            ]
        ]
    )

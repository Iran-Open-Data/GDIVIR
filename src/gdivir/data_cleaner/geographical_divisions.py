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
]


def create_clean_table(year: int) -> pd.DataFrame:
    path = directories.raw_data.joinpath("geographical_divisions", f"{year}.csv")
    table = pd.read_csv(path, dtype=str)

    if "ID" in table.columns:
        table = table.dropna(subset="ID")

    table = table.fillna("")
    if "Village_ID" not in table.columns:
        table["Village_ID"] = ""

    common.extract_ids_from_long_id(table, dataset="geographical_divisions", year=year)
    _create_rural_district_or_city_id(table)
    if (year >= 1365) and (year <= 1385):
        table["Village_ID"] = table["Village_ID"].str.slice(3)
    common.create_long_id(table)
    _create_village_name(table)
    common.create_rural_district_or_city_name(table)
    common.create_region_type_column(table)
    common.set_region_type_labels(table)
    table = _apply_adhoc_editions(table, year)
    table = table.replace("", None)
    table = table.dropna(subset="ID")
    table = table.sort_values("ID")

    table["Year"] = year
    table = table.loc[:, CLEAN_TABLE_COLUMNS]
    assert isinstance(table, pd.DataFrame)
    return table


def _create_rural_district_or_city_id(table: pd.DataFrame) -> None:
    if "Rural_District_or_City_ID" in table.columns:
        return
    if "Rural_District_ID" not in table.columns:
        raise ValueError

    table["Rural_District_or_City_ID"] = (
        table["Rural_District_ID"]
        .mask(
            table["Rural_District_ID"].eq(""),
            table["City_ID"],
        )
    ) 


def _create_village_name(table: pd.DataFrame) -> None:
    if "Village_Name" in table.columns:
        return
    table["Village_Name"] = (
        table["Region_Name"]
        .where(table["Region_Type"].isin(["6", "8"]), "")
    )


def _apply_adhoc_editions(table: pd.DataFrame, year: int) -> pd.DataFrame:
    if year == 1355:
        table.loc[
            table["Province_ID"].eq("03") & table["County_ID"].eq("11"),
            "County_Name",
        ] = "مغان"
    if year == 1372:
        table = _remove_redundant_characters(table, "Province_Name")
        table = _remove_redundant_characters(table, "County_Name")
        table = _remove_redundant_characters(table, "District_Name")
        table = _remove_redundant_characters(table, "Rural_District_or_City_Name")
        table = _remove_redundant_characters(table, "Village_Name")
    if year == 1388:
        table.loc[
            table["Rural_District_or_City_ID"].eq("2337"),
            "ID",
        ] = "0726022337"
        table.loc[
            table["Rural_District_or_City_ID"].eq("2337"),
            "County_ID",
        ] = "26"
        table.loc[
            table["Rural_District_or_City_ID"].eq("2337"),
            "District_ID",
        ] = "02"
    if year == 1390:
        table = _remove_fake_cities(table)
    return table


def _remove_redundant_characters(table: pd.DataFrame, column: str) -> pd.DataFrame:
    table.loc[:, column] = table[column].str.replace("\\sط$", "", regex=True)
    table.loc[:, column] = table[column].str.replace("\\sپ$", "", regex=True)
    return table


def _remove_fake_cities(table: pd.DataFrame) -> pd.DataFrame:
    id_columns = ["Province_ID", "County_ID", "District_ID"]
    fake_city_ids = (
        table
        .loc[lambda df: df["Region_Type"].eq("City")]
        .loc[:, id_columns + ["Rural_District_or_City_Name", "ID"]]
        .merge(
            table
            .loc[lambda df: df["Region_Type"].isin(["Regular_Village", "Block_Village"])]
            .loc[:, id_columns + ["Village_Name"]],
            left_on = id_columns + ["Rural_District_or_City_Name"],
            right_on = id_columns + ["Village_Name"],
        )
        .loc[:, "ID"]
        .drop_duplicates()
    )

    table = table.loc[
        lambda df:
        (- df["ID"].isin(fake_city_ids)) |
        df["ID"].isin(["2245", "2276", "2569", "2604"])
    ]
    return table

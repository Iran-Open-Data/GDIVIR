import pandas as pd

from ..utils import metadata


def extract_ids_from_long_id(table: pd.DataFrame, dataset: str, year: int) -> None:
    if "County_ID" in table.columns:
        return
    if "ID" not in table.columns:
        raise ValueError

    table_metadata = metadata[dataset].get_metadata_version("tables", year)
    for _id, code_range in table_metadata["id"].items():
        table[_id] = table["ID"].str.slice(*code_range)


def create_long_id(table: pd.DataFrame) -> None:
    id_columns = [
        "Province_ID",
        "County_ID",
        "District_ID",
        "Rural_District_or_City_ID",
        "Village_ID",
    ]

    table["ID"] = table[id_columns].sum("columns") # type: ignore


def create_rural_district_or_city_name(table: pd.DataFrame) -> None:
    if "Rural_District_or_City_Name" in table.columns:
        return

    if "City_Name" in table.columns:
        table["Rural_District_or_City_Name"] = (
            table["Rural_District_Name"]
            .mask(
                table["Rural_District_Name"].eq(""),
                table["City_Name"],
            )
        )
    elif ("Region_Type" in table.columns) and ("Region_Name" in table.columns):
        table["Rural_District_or_City_Name"] = (
            table["Rural_District_Name"]
            .mask(
                table["Rural_District_Name"].eq("") & table["Region_Type"].eq("5"),
                table["Region_Name"],
            )
        )
    else:
        raise ValueError


def create_region_type_column(table: pd.DataFrame) -> None:
    if "Region_Type" in table.columns:
        return
    
    table["Region_Type"] = ""
    if "DIAG" in table.columns:
        table.loc[table["DIAG"].ne(""), "Region_Type"] = "8"
    table.loc[
        (
            table["Village_Name"].ne("") |
            table["Village_ID"].ne("") 
        )&
        table["Region_Type"].eq(""),
        "Region_Type",
    ] = "6"
    table.loc[table["City_Name"].ne(""), "Region_Type"] = "5"
    table.loc[
        table["Village_Name"].eq("") &
        (
            table["Rural_District_Name"].ne("") |
            table["Rural_District_or_City_ID"].ne("")
        ) &
        table["Region_Type"].eq(""),
        "Region_Type",
    ] = "4"
    table.loc[table["County_Name"].eq(""), "Region_Type"] = "1"
    table.loc[
        table["District_Name"].eq("") &
        table["District_ID"].eq("") &
        table["Region_Type"].eq(""),
        "Region_Type",
    ] = "2"
    table.loc[table["Region_Type"].eq(""), "Region_Type"] = "3"


def set_region_type_labels(table: pd.DataFrame) -> None:
    region_type_dtype = pd.CategoricalDtype(
        categories = [
            "Province",
            "County",
            "District",
            "Rural_District",
            "City",
            "City_District",
            "City_Virtual_District",
            "Regular_Village",
            "Block_Village",
            "Non_Resident",
        ],
    )

    table["Region_Type"] = (
        table["Region_Type"]
        .replace(
            {
                "1": "Province",
                "2": "County",
                "3": "District",
                "4": "Rural_District",
                "5": "City",
                "6": "Regular_Village",
                "8": "Block_Village",
            }
        )
        .astype(region_type_dtype)
    )
    table.loc[
        table["Region_Type"].eq("City") &
        table["Rural_District_or_City_Name"].str.contains("\\d", regex=True),
        "Region_Type",
    ] = "City_District"
    table.loc[
        table["Region_Type"].eq("City") &
        table["Rural_District_or_City_Name"].str.contains("منطقه"),
        "Region_Type",
    ] = "City_District"
    table.loc[
        table["Region_Type"].eq("City") &
        (
            table["Village_Name"].ne("") |
            table["Village_ID"].ne("")
        ),
        "Region_Type",
    ] = "City_Virtual_District"
    table.loc[table["District_ID"].eq("99"), "Region_Type"] = "Non_Resident"

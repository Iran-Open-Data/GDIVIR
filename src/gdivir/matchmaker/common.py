from typing import Literal

import pandas as pd

from ..utils import _RegionType, directories, metadata


def add_population(geo_data: pd.DataFrame, census_year: int) -> pd.DataFrame:
    cnesus_data = pd.read_parquet(
        directories.census_results,
        filters=[("Year", "=", census_year)],
    )
    city_part = (
        geo_data
        .loc[_filter_cities, ["Rural_District_or_City_ID"]]
        .join(
            cnesus_data
            .loc[_filter_cities]
            .set_index(["Rural_District_or_City_ID"])
            .loc[:, ["Household_Count", "Population"]]
            .rename(lambda name: f"City_{name}", axis="columns")
            ,
            on="Rural_District_or_City_ID",
            validate="1:1",
        )
        .drop(columns=["Rural_District_or_City_ID"])
    )
    rural_part = (
        geo_data
        .loc[_filter_villages, ["Village_ID"]]
        .join(
            cnesus_data
            .loc[_filter_villages]
            .set_index(["Village_ID"])
            .loc[:, ["Household_Count", "Population"]]
            .rename(lambda name: f"Village_{name}", axis="columns")
            .dropna()
            ,
            on="Village_ID",
            validate="1:1",
        )
        .drop(columns=["Village_ID"])
    )
    return (
        pd.concat(
            [
                geo_data,
                city_part,
                rural_part,
            ],
            axis="columns",
            join="outer",
        )
        .assign(
            Household_Count=_calculate_total_household_count,
            Population=_calculate_total_population,
        )
        .drop(columns=[
            "City_Household_Count",
            "City_Population",
            "Village_Household_Count",
            "Village_Population",
        ])
    )


def _filter_villages(df: pd.DataFrame) -> pd.Series:
    return df["Region_Type"].isin(["Regular_Village", "Block_Village"])


def _filter_cities(df: pd.DataFrame) -> pd.Series:
    return df["Region_Type"].isin(["City"])


def _calculate_total_household_count(df: pd.DataFrame) -> pd.Series:
    return (
        df[["City_Household_Count", "Village_Household_Count"]]
        .fillna(0)
        .sum(axis="columns") # type: ignore
        .replace(0, None)
    )


def _calculate_total_population(df: pd.DataFrame) -> pd.Series:
    return (
        df[["City_Population", "Village_Population"]]
        .fillna(0)
        .sum(axis="columns") # type: ignore
        .replace(0, None)
    )


def create_village_table(year: int) -> pd.DataFrame:
    return (
        pd.read_parquet(
            directories.geographical_divisions,
            filters=[("Year", "=", year)],
        )
        .loc[_filter_villages]
        .assign(
            County_ID=lambda df: df[["Province_ID", "County_ID"]].sum(axis="columns"),
        )
        .drop_duplicates("Village_ID")
        .pipe(
            add_population,
            metadata.census_results.get_nearest_year(year)
        )
    )


def create_city_table(year: int) -> pd.DataFrame:
    return (
        pd.read_parquet(
            directories.geographical_divisions,
            filters=[("Year", "=", year)],
        )
        .loc[_filter_cities]
        .assign(
            City_ID=lambda df: df["Rural_District_or_City_ID"],
            County_ID=lambda df: df[["Province_ID", "County_ID"]].sum(axis="columns"),
        )
        .pipe(
            add_population,
            metadata.census_results.get_nearest_year(year)
        )
    )


def _create_village_transformation_table(
    year: int,
    region_type: Literal["Province", "County", "District", "Rural_District"],
) -> pd.DataFrame:
    previous_year = metadata.geographical_divisions.get_previous_year(year)
    return (
        pd.merge(
            create_village_table(previous_year),
            create_village_table(year),
            how="outer",
            on="Village_ID",
            suffixes=("_Old", "_New"),
            validate="1:1",
            indicator=True,
        )        
        .pipe(_create_transformation_table, region_type)
    )


def _create_city_transformation_table(
    year: int,
    region_type: Literal["Province", "County", "District", "Rural_District"],
) -> pd.DataFrame:
    previous_year = metadata.geographical_divisions.get_previous_year(year)
    return (
        pd.merge(
            create_city_table(previous_year),
            create_city_table(year),
            how="outer",
            on="City_ID",
            suffixes=("_Old", "_New"),
            validate="1:1",
            indicator=True,
        )
        .pipe(_create_transformation_table, region_type)
    )


def _create_transformation_table(df: pd.DataFrame, region_type: _RegionType) -> pd.DataFrame:
    return (
        df       
        .groupby([f"{region_type}_ID_New", f"{region_type}_ID_Old"], as_index=False)
        .aggregate(
            {
                "ID_New": "count",
                "Household_Count_New": "sum",
                "Population_New": "sum",
            }
        )
        .rename(
            columns={
                f"{region_type}_ID_New": f"New_{region_type}_ID",
                f"{region_type}_ID_Old": f"Old_{region_type}_ID",
                "ID_New": "Shared_Region_Count",
                "Household_Count_New": "Shared_Household_Count",
                "Population_New": "Shared_Population",
            }
        )
    )


def create_population_transformation_table(year: int, region_type: _RegionType) -> pd.DataFrame:
    return (
        pd.merge(
            _create_city_transformation_table(year, region_type),
            _create_village_transformation_table(year, region_type),
            on=[f"New_{region_type}_ID", f"Old_{region_type}_ID"],
            how="outer",
            suffixes=("_City", "_Village"),
        )
        .fillna(0)
        .assign(
            Shared_Household_Count = lambda df:
            df["Shared_Household_Count_City"] +
            df["Shared_Household_Count_Village"]
            ,
            Shared_Population = lambda df:
            df["Shared_Population_City"] +
            df["Shared_Population_Village"]
            ,
            Population_Share = lambda df:
            df.groupby(f"New_{region_type}_ID")["Shared_Population"]
            .transform(lambda s: s / s.sum() * 100)
        )
    )


def create_many_to_one_mapping_documentation(year: int, region_type: _RegionType) -> pd.DataFrame:
    return (
        create_population_transformation_table(year, region_type)
        .sort_values([f"New_{region_type}_ID", "Population_Share"], ascending=[True, False])
        .assign(
            Selected = lambda df:
            - df[f"New_{region_type}_ID"].duplicated(keep="first")
        )
    )


def create_many_to_one_mapping(year: int, region_type: _RegionType) -> dict:
    return(
        create_many_to_one_mapping_documentation(year, region_type)
        .loc[lambda df: df["Selected"]]
        .set_index(f"New_{region_type}_ID")
        .loc[:, f"Old_{region_type}_ID"]
        .to_dict()
    )

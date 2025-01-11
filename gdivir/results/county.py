from pathlib import Path

import pandas as pd

from ..matchmaker import create_many_to_one_mapping
from ..utils import metadata


def export_many_to_one_county_matching() -> None:
    mapping_string = ""
    for year in metadata.geographical_divisions.years[3:]:
        mapping = create_many_to_one_mapping(year, "County")
        mapping_string += f"{year}:\n"
        mapping_string += "\n".join(
            [
                f"  '{key}': '{value}'"
                for key, value in mapping.items()
                if key != value
            ]
        )
        mapping_string += "\n\n"
    file_path = Path("results", "county_many_to_one_mapping.yaml")
    file_path.parent.mkdir(exist_ok=True)
    with file_path.open(mode="w") as file:
        file.write(mapping_string)


def export_many_to_one_mapping_table() -> None:
    mapping_parts: list[pd.DataFrame] = []
    for year in metadata.geographical_divisions.years[3:]:
        mapping_parts.append(
            pd.Series(
                create_many_to_one_mapping(year, "County"),
                name=metadata.geographical_divisions.get_previous_year(year),
            )
            .rename_axis(year)
            .reset_index()
        )

    mapping = mapping_parts[0]
    for mapping_part in mapping_parts[1:]:
        mapping = pd.merge(mapping, mapping_part, how="outer")
    mapping = mapping.loc[:, sorted(mapping.columns)]
    mapping = mapping.sort_values(mapping.columns.to_list()[::-1])
    file_path = Path("results", "county_many_to_one_mapping_table.csv")
    file_path.parent.mkdir(exist_ok=True)
    mapping.to_csv(file_path, index=False)

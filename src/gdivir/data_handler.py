from pathlib import Path
import shutil
from io import BytesIO
from typing import Literal
import warnings
from zipfile import ZipFile

import pandas as pd

from . import data_cleaner
from .utils import download, metadata, _Dataset


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="openpyxl.worksheet.header_footer"
)


def setup_data(dataset: _Dataset = "geographical_divisions") -> None:
    download_original_files("all", dataset)
    extract_raw_files("all", dataset)
    create_clean_dataset(dataset)


def download_original_files(
    years: int | list[int] | Literal["all"],
    dataset: _Dataset = "geographical_divisions",
) -> None:
    years = _parse_years(years, dataset)
    data_set_metadata = metadata.raw_files[dataset]
    for year in years:
        original_file_directory = Path("data", "original", dataset, str(year))
        for directory_name, url in data_set_metadata[year].items():
            directory = original_file_directory.joinpath(directory_name)
            download(url, directory)
    

def extract_raw_files(
    years: int | list[int] | Literal["all"],
    dataset: _Dataset = "geographical_divisions",
) -> None:
    years = _parse_years(years, dataset)
    raw_directory = Path("data", "raw", dataset)
    raw_directory.mkdir(parents=True, exist_ok=True)
    for year in years:
        table = _extract_data_from_excel(year, dataset)
        data_cleaner.apply_general_cleaning(table)
        table.to_csv(raw_directory.joinpath(f"{year}.csv"), index=False)


def create_clean_dataset(dataset: _Dataset = "geographical_divisions") -> None:
    table_list = []
    if dataset == "geographical_divisions":
        create_clean_table = data_cleaner.geographical_divisions.create_clean_table
    elif dataset == "census_results":
        create_clean_table = data_cleaner.census_results.create_clean_table
    else:
        raise ValueError

    for year in metadata.raw_files[dataset].keys():
        table_list.append(create_clean_table(year))
    table = pd.concat(table_list, ignore_index=True)

    clean_data_path = Path("data", "cleaned", f"{dataset}.parquet")
    clean_data_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(clean_data_path, ignore_errors=True)
    table.to_parquet(clean_data_path, partition_cols=["Year"])


def _parse_years(
    years: int | list[int] | Literal["all"],
    data_set: str,
) -> list[int]:
    if isinstance(years, int):
        years = [years]
    elif years == "all":
        years = list(metadata.raw_files[data_set].keys())
    return years


def _extract_data_from_excel(year: int, dataset: _Dataset) -> pd.DataFrame:
    data_set_directory = Path("data", "original", dataset)
    if dataset == "geographical_divisions":
        directories = [data_set_directory.joinpath(str(year), "division")]
    elif dataset == "census_results":
        directories = data_set_directory.joinpath(str(year)).iterdir()
    else:
        raise ValueError
    excel_files = [_read_excel_file(path) for path in directories]
    table_metadata = metadata[dataset].get_metadata_version("tables", year)
    
    tables: list[pd.DataFrame] = []
    for excel_file in excel_files:
        for sheet_name in excel_file.sheet_names:
            sheet = _open_excel_sheet(excel_file, sheet_name, table_metadata)
            if sheet is None:
                continue
            tables.append(sheet)
    table = pd.concat(tables, ignore_index=True)
    return table


def _read_excel_file(directory_path: Path) -> pd.ExcelFile:
    division_file = list(directory_path.glob("*"))[0]
    if division_file.suffix.lower() in [".xls", ".xlsx"]:
        excel_file = pd.ExcelFile(division_file)
    elif division_file.suffix.lower() == ".zip":
        excel_file = _extract_excel_from_zip(division_file)
    else:
        raise ValueError
    return excel_file


def _extract_excel_from_zip(file_path: Path) -> pd.ExcelFile:
    with ZipFile(file_path) as zip_file:
        file_list = [file.filename for file in zip_file.filelist if file.file_size > 0]
        assert len(file_list) == 1
        excel_bytes = BytesIO(zip_file.read(file_list[0]))
    excel_file = pd.ExcelFile(excel_bytes)
    return excel_file


def _open_excel_sheet(
        excel_file: pd.ExcelFile,
        sheet_name: str | int,
        table_metadata: dict
) -> pd.DataFrame | None:
    try:
        table = excel_file.parse(
            sheet_name,
            header=None,
            skiprows=table_metadata.get("skiprows", 1),
            usecols=table_metadata.get("usecols", None),
            converters={i: str for i in range(len(table_metadata["columns"]))},
        )
        table.columns = table_metadata["columns"]
    except (ValueError, IndexError):
        return None
    columns = table.columns.to_list()
    columns = [column for column in columns if "_drop_" not in column.lower()]
    if table_metadata.get("reverse", False):
        columns.reverse()
    table = table[columns]
    assert isinstance(table, pd.DataFrame)
    return table

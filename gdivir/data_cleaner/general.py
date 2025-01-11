import pandas as pd


INVISIBLE_CHARS = [
    chr(8203),
    chr(173),
    chr(8207),
    chr(8236),
    chr(8234),
    chr(65279)
]

UNWANTED_SYMBOLS = [
    "\n",
    "\r",
    "\t",
    "…",
    "ـ",
    "_",
    "\\-",
    "•",
    "\\*",
    "`",
    "\"",
    "\'",
    "«",
    "»",
    ".",
    ",",
    ";",
    ":",
]


def apply_general_cleaning(table: pd.DataFrame) -> None:
    farsi_columns = _get_farsi_columns(table)
    table.loc[:, farsi_columns] = table.loc[:, farsi_columns].apply(_clean_farsi_text)
    id_columns = _get_id_columns(table)
    table.loc[:, id_columns] = table.loc[:, id_columns].apply(_clean_ids)
    numeric_columns = _get_numeric_columns(table)
    table.loc[:, numeric_columns] = (
        table.loc[:, numeric_columns]
        .apply(_clean_numeric_columns)
    )


def _clean_farsi_text(s: pd.Series) -> pd.Series:
    s = _replace_arabic_characters(s)

    # Replace Zero Width Non-Joiner ('\u200c') with a space
    s = s.str.replace(chr(8204), " ")

    # Remove other invisible and unwanted characters
    pattern = "[" + "".join(INVISIBLE_CHARS + UNWANTED_SYMBOLS) + "]"
    s = s.str.replace(pattern, "", regex=True)

    # Normalize spaces: replace all multi-space occurrences with a single space
    s = s.str.replace("\\s+", " ", regex=True)
    s = s.str.replace("\\( ", "(", regex=True)
    s = s.str.replace(" \\)", ")", regex=True)
    s = s.str.strip()

    return s


def _replace_arabic_characters(s: pd.Series) -> pd.Series:
    for old_char, new_char in [
        (chr(1610), chr(1740)), # ي -> ی
        (chr(1574), chr(1740)), # ئ -> ی
        (chr(1609), chr(1740)), # ى -> ی
        (chr(1571), chr(1575)), # أ -> ا
        (chr(1573), chr(1575)), # إ -> ا
        (chr(1572), chr(1608)), # ؤ -> و
        (chr(1603), chr(1705)), # ك -> ک
        (chr(1728), chr(1607)), # ۀ -> ه
        (chr(1577), chr(1607)), # ة -> ه
    ]:
        s = s.str.replace(old_char, new_char)
    return s


def _get_farsi_columns(df: pd.DataFrame) -> list:
    return [column for column in df.columns if "Name" in column]


def _clean_ids(s: pd.Series) -> pd.Series:
    return s.str.replace("\\D", "", regex=True)


def _get_id_columns(df: pd.DataFrame) -> list:
    return [
        column for column in df.columns
        if ("ID" in column) or (column in ["Region_Type", "DIAG"])
    ]


def _clean_numeric_columns(s: pd.Series) -> pd.Series:
    return (
        s
        .astype(str)
        .str.extract("(\\d+)").loc[:, 0]
        .replace("", None)
        .astype("Int64")
    )


def _get_numeric_columns(df: pd.DataFrame) -> list:
    return [
        column for column in df.columns
        if column in ["Household_Count", "Population"]
    ]

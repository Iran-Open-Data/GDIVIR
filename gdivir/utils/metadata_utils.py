def find_metadata_version(versions: dict, year: int) -> dict:
    last_version = 0
    for version_year in versions.keys():
        if (version_year <= year) & (last_version <= year):
            last_version = version_year
    assert last_version != 0
    return versions[last_version]

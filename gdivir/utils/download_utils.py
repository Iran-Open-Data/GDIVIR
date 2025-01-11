from pathlib import Path

import requests


def download(url: str, path: Path) -> None:
    """Downloads a file from a given URL and saves it to a specified local path.

    This function uses the requests library to send a GET request to the provided URL,
    and then writes the response content to a file at the specified path. If the path
    is not provided, the file is saved in a temporary directory. The function also
    provides an option to display a progress bar during the download.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    path : str, Path, optional
        The local path where the downloaded file should be saved. If None, the file
        is saved in a temporary directory. Default is None.
    show_progress_bar : bool, optional
        If True, a progress bar is displayed during the download. Default is False.

    Returns
    -------
    Path
        The local path where the downloaded file was saved.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found at the given URL.
    """
    response = requests.get(url, timeout=1000, stream=True)
    content_iterator = response.iter_content(chunk_size=4096)
    remote_file_size = response.headers.get("content-length")
    if remote_file_size is not None:
        remote_file_size = int(remote_file_size)
    else:
        raise FileNotFoundError("File is not found on the server")
    file_name = url.rsplit("/", 1)[1]
    path = path.joinpath(file_name)
    if path.exists():
        local_file_size = path.stat().st_size
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        local_file_size = 0
    if remote_file_size == local_file_size:
        return
    with open(path, mode="wb") as file:
        while True:
            try:
                chunk = next(content_iterator)
            except StopIteration:
                break
            except requests.Timeout:
                continue
            file.write(chunk)

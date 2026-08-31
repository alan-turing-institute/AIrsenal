"""Downloading a file over HTTP, resumably."""

from pathlib import Path

from curl_cffi import requests

from airsenal.remote.errors import RemoteError

DEFAULT_CHUNK_SIZE = 1024 * 1024


def download_with_resume(
    url: str,
    dest: Path,
    attempts: int = 5,
    timeout: float = 30.0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    """
    Download `url` to `dest`, picking up where an interrupted attempt left off.

    Asks for the bytes after whatever is already on disk, so a retry over a flaky
    connection does not start the file again. A server that ignores the `Range`
    header answers 200 rather than 206, in which case the partial file is
    discarded and written from the start.

    Raises:
        RemoteError: If every attempt fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for attempt in range(1, attempts + 1):
        existing = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
        resp = None
        try:
            # Inside the try: a refused connection, a DNS failure or a timeout
            # raises here rather than from the body, and outside it that escaped
            # as a raw curl_cffi error - unretried, and past every downstream
            # `except RemoteError`.
            resp = session.get(
                url,
                headers=headers,
                stream=True,
                timeout=timeout,
            )
            resp.raise_for_status()

            # If server ignored Range (status 200), restart file from scratch.
            if existing > 0 and resp.status_code == 200:
                mode = "wb"
            else:
                mode = "ab" if existing > 0 else "wb"

            with open(dest, mode) as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

            return dest

        except requests.exceptions.RequestException as e:
            if attempt == attempts:
                msg = f"Failed to download {url} after {attempts} attempts"
                raise RemoteError(msg) from e
        finally:
            if resp is not None:
                resp.close()

    return dest

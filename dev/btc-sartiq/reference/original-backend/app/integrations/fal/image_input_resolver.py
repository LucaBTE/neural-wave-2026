import base64
import mimetypes
from urllib.parse import urlparse
from urllib.request import urlopen

from fastapi import HTTPException

# this resolves the input image url to a format that can be used by fal.
# if a Data URI is provided, it is returned as is.


def resolve_image_input(input_image_url: str) -> str:
    source = input_image_url.strip()
    if not source:
        raise HTTPException(status_code=400, detail="input image url is required")

    if source.startswith("data:"):
        return source

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return source

    raise HTTPException(status_code=400, detail="unsupported input image source")


def build_data_uri_from_remote_url(remote_url: str) -> str:
    normalized_url = remote_url.strip()
    parsed = urlparse(normalized_url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="input image url must be http or https")

    try:
        with urlopen(normalized_url, timeout=8) as response:
            content = response.read()
            mime_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip()
    except Exception:
        raise HTTPException(status_code=400, detail="cannot download input image")

    if not mime_type:
        mime_type, _ = mimetypes.guess_type(parsed.path)
    if not mime_type:
        mime_type = "application/octet-stream"

    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

import os

import fal_client
from fastapi import HTTPException

DEFAULT_TIMEOUT_SECONDS = 120


def run_model(model_id: str, payload: dict) -> dict:
    api_key = os.getenv("FAL_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="FAL_API_KEY is not in .env file yet")

    normalized_model_id = model_id.strip()
    if not normalized_model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    os.environ["FAL_KEY"] = api_key

    try:
        return fal_client.subscribe(
            normalized_model_id,
            arguments=payload,
            with_logs=False,
            client_timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="fal request timed out while waiting for result")
    except Exception as exc:
        detail = str(exc).strip() or "unexpected error calling fal"
        raise HTTPException(status_code=502, detail=f"fal request failed: {detail}")

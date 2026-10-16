from __future__ import annotations

import base64
from io import BytesIO
from urllib.request import urlopen

from fastapi import HTTPException
from PIL import Image, ImageChops

from app.integrations.fal.fal_client import run_model
from app.integrations.fal.image_input_resolver import resolve_image_input
from app.services.storage_service import save_project_bytes


class FalModelGateway:
    def run(self, model_id: str, payload: dict[str, object]) -> dict:
        return run_model(model_id, payload)

    def resolve_image(self, image_url: str) -> str:
        return resolve_image_input(image_url)

    def extract_first_image_url(self, response: object) -> str | None:
        if not isinstance(response, dict):
            return None

        for response_part in (response, response.get("data")):
            if not isinstance(response_part, dict):
                continue

            image = response_part.get("image")
            if isinstance(image, dict):
                value = image.get("url")
                if isinstance(value, str) and value.strip():
                    return value.strip()

            images = response_part.get("images")
            if isinstance(images, list) and images:
                image_url = self._read_image_item(images[0])
                if image_url:
                    return image_url

            for key in ("mask", "mask_image", "mask_url", "output_url", "image_url", "url"):
                value = response_part.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, dict):
                    nested_url = value.get("url")
                    if isinstance(nested_url, str) and nested_url.strip():
                        return nested_url.strip()

        return None

    def merge_mask_items(
        self,
        items: list[object],
        *,
        project_id: int | None = None,
        apply_mask: bool = True,
        output_name: str = "merged-mask.png",
    ) -> str | None:
        mask_urls = [mask_url for item in items if (mask_url := self._read_image_item(item))]
        if not mask_urls:
            return None

        if len(mask_urls) == 1:
            return mask_urls[0]

        merged_image: Image.Image | None = None
        base_size: tuple[int, int] | None = None

        for mask_url in mask_urls:
            current_image = self._load_image(mask_url)
            current_image = current_image.convert("RGBA" if apply_mask else "L")
            if merged_image is None:
                merged_image = current_image
                base_size = current_image.size
                continue

            if base_size and current_image.size != base_size:
                current_image = current_image.resize(base_size)

            merged_image = (
                Image.alpha_composite(merged_image, current_image)
                if apply_mask
                else ImageChops.lighter(merged_image, current_image)
            )

        if merged_image is None:
            return None

        if project_id is not None:
            return save_project_bytes(project_id, self._image_to_png_bytes(merged_image), output_name)

        return self._image_to_data_uri(merged_image)

    def _read_image_item(self, item: object) -> str | None:
        if isinstance(item, str) and item.strip():
            return item.strip()
        if isinstance(item, dict):
            value = item.get("url")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _load_image(self, source: str) -> Image.Image:
        try:
            with urlopen(source) as response:
                image = Image.open(BytesIO(response.read()))
                image.load()
                return image
        except Exception as exc:
            raise HTTPException(status_code=502, detail="failed to download segmentation mask") from exc

    def _image_to_data_uri(self, image: Image.Image) -> str:
        with BytesIO() as output:
            image.save(output, format="PNG")
            encoded = base64.b64encode(output.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}"

    def _image_to_png_bytes(self, image: Image.Image) -> bytes:
        with BytesIO() as output:
            image.save(output, format="PNG")
            return output.getvalue()

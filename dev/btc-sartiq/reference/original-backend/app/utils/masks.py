from __future__ import annotations

import base64
from io import BytesIO
from typing import Literal
from urllib.request import urlopen

from fastapi import HTTPException
from PIL import Image, ImageDraw, ImageFilter

from app.services.storage_service import save_project_bytes


def build_convex_hull_mask(
    mask_image_url: str,
    project_id: int | None = None,
    mode: Literal["simple", "medium", "rectangle"] = "medium",
) -> str:
    mask_image = _load_image(mask_image_url).convert("L")
    binary_mask = mask_image.point(lambda pixel: 255 if pixel > 0 else 0)
    foreground_points = _collect_foreground_points(binary_mask)
    hull_mask = Image.new("L", binary_mask.size, 0)

    if not foreground_points:
        return _store_image(hull_mask, project_id)

    draw = ImageDraw.Draw(hull_mask)
    if mode == "rectangle":
        min_x = min(point[0] for point in foreground_points)
        min_y = min(point[1] for point in foreground_points)
        max_x = max(point[0] for point in foreground_points)
        max_y = max(point[1] for point in foreground_points)
        draw.rectangle((min_x, min_y, max_x, max_y), fill=255)
    else:
        hull_points = _convex_hull(foreground_points)
        if len(hull_points) == 1:
            x, y = hull_points[0]
            draw.point((x, y), fill=255)
        elif len(hull_points) == 2:
            draw.line(hull_points, fill=255, width=1)
        else:
            draw.polygon(hull_points, fill=255)

        if mode == "medium":
            hull_mask = _expand_mask(hull_mask, 24)

    return _store_image(hull_mask, project_id)


def _load_image(source: str) -> Image.Image:
    try:
        with urlopen(source) as response:
            image = Image.open(BytesIO(response.read()))
            image.load()
            return image
    except Exception as exc:
        raise HTTPException(status_code=502, detail="failed to download segmentation mask") from exc


def _collect_foreground_points(mask_image: Image.Image) -> list[tuple[int, int]]:
    width, height = mask_image.size
    pixels = mask_image.load()
    points: list[tuple[int, int]] = []

    for y in range(height):
        for x in range(width):
            if pixels[x, y] > 0:
                points.append((x, y))

    return points


def _convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    unique_points = sorted(set(points))
    if len(unique_points) <= 1:
        return unique_points

    def cross(origin: tuple[int, int], point_a: tuple[int, int], point_b: tuple[int, int]) -> int:
        return (point_a[0] - origin[0]) * (point_b[1] - origin[1]) - (point_a[1] - origin[1]) * (
            point_b[0] - origin[0]
        )

    lower: list[tuple[int, int]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[int, int]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _expand_mask(mask_image: Image.Image, expand_px: int) -> Image.Image:
    kernel_size = max(1, expand_px * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return mask_image.filter(ImageFilter.MaxFilter(kernel_size))


def _image_to_data_uri(image: Image.Image) -> str:
    with BytesIO() as output:
        image.save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


def _image_to_png_bytes(image: Image.Image) -> bytes:
    with BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()


def _store_image(image: Image.Image, project_id: int | None) -> str:
    if project_id is not None:
        return save_project_bytes(project_id, _image_to_png_bytes(image), "segmentation-convex-hull-mask.png")

    return _image_to_data_uri(image)

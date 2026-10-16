from __future__ import annotations

from app.models_registry.base import (
    AdditionalSettingDefinitionData,
    AdditionalSettingOptionDefinition,
    ProcessModelDefinition,
)
from app.models_registry.fal_adapters import (
    FalBriaGenFillAdapter,
    FalFinegrainEraserMaskAdapter,
    FalFluxFillAdapter,
    FalMoondreamSegmentationAdapter,
    FalSamSegmentationAdapter,
)
from app.models_registry.fal_gateway import FalModelGateway

fal_gateway = FalModelGateway()

SEGMENTATION_MODELS: dict[str, ProcessModelDefinition] = {
    "sam3": ProcessModelDefinition(
        key="sam3",
        label="SAM 3.1",
        process_type="segment_from_prompt",
        provider="fal",
        provider_model_id="fal-ai/sam-3-1/image",
        adapter=FalSamSegmentationAdapter(fal_gateway, "fal-ai/sam-3-1/image"),
        default=True,
        additional_settings=(
            AdditionalSettingDefinitionData(
                key="apply_mask",
                label="Apply mask",
                type="boolean",
                description="Return the image with the segmentation mask already applied.",
                default_value=False,
            ),
            AdditionalSettingDefinitionData(
                key="return_multiple_masks",
                label="Multiple masks",
                type="boolean",
                description="Ask the model to return multiple candidate masks for the same prompt.",
                default_value=False,
            ),
            AdditionalSettingDefinitionData(
                key="max_masks",
                label="Max masks",
                type="integer",
                description="Maximum number of masks returned when multiple masks are enabled.",
                depends_on_key="return_multiple_masks",
                depends_on_value=True,
                default_value=3,
                min_value=1,
                max_value=10,
                step=1,
            ),
            AdditionalSettingDefinitionData(
                key="include_mask_scores",
                label="Include scores",
                type="boolean",
                description="Include confidence scores for the returned masks.",
                default_value=False,
            ),
            AdditionalSettingDefinitionData(
                key="include_bboxes",
                label="Include boxes",
                type="boolean",
                description="Include bounding boxes for each mask when available.",
                default_value=False,
            ),
        ),
    ),
    "moondream3": ProcessModelDefinition(
        key="moondream3",
        label="Moondream 3 Preview",
        process_type="segment_from_prompt",
        provider="fal",
        provider_model_id="fal-ai/moondream3-preview/segment",
        adapter=FalMoondreamSegmentationAdapter(fal_gateway, "fal-ai/moondream3-preview/segment"),
        additional_settings=(
            AdditionalSettingDefinitionData(
                key="preview",
                label="Preview mask",
                type="boolean",
                description="Return a binary mask preview image together with the segmentation result.",
                default_value=True,
            ),
            AdditionalSettingDefinitionData(
                key="temperature",
                label="Temperature",
                type="number",
                description="Sampling temperature for the segmentation model.",
                default_value=0.0,
                min_value=0.0,
                max_value=2.0,
                step=0.1,
            ),
            AdditionalSettingDefinitionData(
                key="top_p",
                label="Top P",
                type="number",
                description="Nucleus sampling probability mass between 0 and 1.",
                default_value=1.0,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
            ),
            AdditionalSettingDefinitionData(
                key="max_tokens",
                label="Max tokens",
                type="integer",
                description="Maximum number of decoding tokens used by the model.",
                default_value=1024,
                min_value=1,
                max_value=4096,
                step=1,
            ),
        ),
    ),
}

REMOVAL_MODELS: dict[str, ProcessModelDefinition] = {
    "finegrain-eraser": ProcessModelDefinition(
        key="finegrain-eraser",
        label="Finegrain Eraser",
        process_type="remove_with_mask",
        provider="fal",
        provider_model_id="fal-ai/finegrain-eraser/mask",
        adapter=FalFinegrainEraserMaskAdapter(fal_gateway, "fal-ai/finegrain-eraser/mask"),
        default=True,
        additional_settings=(
            AdditionalSettingDefinitionData(
                key="mode",
                label="Mode",
                type="select",
                description="Choose the removal quality mode used by Finegrain Eraser.",
                default_value="standard",
                options=(
                    AdditionalSettingOptionDefinition(value="express", label="Express"),
                    AdditionalSettingOptionDefinition(value="standard", label="Standard"),
                    AdditionalSettingOptionDefinition(value="premium", label="Premium"),
                ),
            ),
        ),
    ),
}

GENERATION_MODELS: dict[str, ProcessModelDefinition] = {
    "flux-fill-pro": ProcessModelDefinition(
        key="flux-fill-pro",
        label="FLUX.1 [pro] Fill",
        process_type="generate_from_prompt",
        provider="fal",
        provider_model_id="fal-ai/flux-pro/v1/fill",
        adapter=FalFluxFillAdapter(fal_gateway, "fal-ai/flux-pro/v1/fill"),
        default=True,
    ),
    "bria-genfill": ProcessModelDefinition(
        key="bria-genfill",
        label="Bria GenFill",
        process_type="generate_from_prompt",
        provider="fal",
        provider_model_id="fal-ai/bria/genfill",
        adapter=FalBriaGenFillAdapter(fal_gateway, "fal-ai/bria/genfill"),
        additional_settings=(
            AdditionalSettingDefinitionData(
                key="negative_prompt",
                label="Negative prompt",
                type="text",
                description="Optional negative prompt forwarded to the Bria fill model.",
            ),
        ),
    ),
}


def get_process_model_registry(process_type: str) -> dict[str, ProcessModelDefinition]:
    if process_type == "segment_from_prompt":
        return SEGMENTATION_MODELS
    if process_type == "remove_with_mask":
        return REMOVAL_MODELS
    if process_type == "generate_from_prompt":
        return GENERATION_MODELS
    return {}

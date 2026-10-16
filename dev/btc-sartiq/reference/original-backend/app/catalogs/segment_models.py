from app.models_registry import SEGMENTATION_MODELS

SEGMENT_MODEL_REGISTRY = SEGMENTATION_MODELS


def get_segment_model_definition(model_key: str):
    return SEGMENT_MODEL_REGISTRY.get(model_key)


def iter_segment_model_definitions():
    return iter(SEGMENT_MODEL_REGISTRY.items())

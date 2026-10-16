from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from fastapi import HTTPException

PrimitiveSettingValue = bool | int | float | str
PrimitiveSettingMap = dict[str, PrimitiveSettingValue]
SettingType = Literal["boolean", "select", "integer", "number", "text"]
ProcessType = Literal["segment_from_prompt", "remove_with_mask", "generate_from_prompt"]


@dataclass(frozen=True)
class AdditionalSettingOptionDefinition:
    value: str
    label: str


@dataclass(frozen=True)
class AdditionalSettingDefinitionData:
    key: str
    label: str
    type: SettingType
    description: str | None = None
    depends_on_key: str | None = None
    depends_on_value: PrimitiveSettingValue | None = None
    default_value: PrimitiveSettingValue | None = None
    options: tuple[AdditionalSettingOptionDefinition, ...] = ()
    min_value: int | float | None = None
    max_value: int | float | None = None
    step: int | float | None = None


@dataclass(frozen=True)
class SegmentModelRequest:
    input_image_url: str
    prompt: str
    project_id: int | None = None
    additional_settings: PrimitiveSettingMap = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationModelRequest:
    input_image_url: str
    mask_image_url: str
    prompt: str | None = None
    project_id: int | None = None
    additional_settings: PrimitiveSettingMap = field(default_factory=dict)


RemovalModelRequest = GenerationModelRequest


class SegmentModelAdapter(Protocol):
    def run(self, request: SegmentModelRequest) -> str:
        ...


class GenerationModelAdapter(Protocol):
    def run(self, request: GenerationModelRequest) -> str:
        ...


class RemovalModelAdapter(Protocol):
    def run(self, request: RemovalModelRequest) -> str:
        ...


@dataclass(frozen=True)
class ProcessModelDefinition:
    key: str
    label: str
    process_type: ProcessType
    provider: str
    provider_model_id: str
    adapter: SegmentModelAdapter | GenerationModelAdapter | RemovalModelAdapter
    supports_text_prompt: bool = True
    default: bool = False
    additional_settings: tuple[AdditionalSettingDefinitionData, ...] = ()


def validate_additional_settings(
    definitions: tuple[AdditionalSettingDefinitionData, ...],
    raw_settings: dict[str, PrimitiveSettingValue] | None,
) -> PrimitiveSettingMap:
    provided = raw_settings or {}
    resolved: PrimitiveSettingMap = {}
    defaults = {item.key: item.default_value for item in definitions}

    for definition in definitions:
        if definition.depends_on_key:
            parent_value = provided.get(definition.depends_on_key, defaults.get(definition.depends_on_key))
            if parent_value != definition.depends_on_value:
                continue

        value = provided.get(definition.key, definition.default_value)
        if value is None:
            continue

        if definition.type == "boolean":
            if not isinstance(value, bool):
                raise HTTPException(status_code=400, detail=f'invalid value for "{definition.key}"')
            resolved[definition.key] = value
            continue

        if definition.type == "select":
            if not isinstance(value, str):
                raise HTTPException(status_code=400, detail=f'invalid value for "{definition.key}"')

            allowed_values = {option.value for option in definition.options}
            if value not in allowed_values:
                raise HTTPException(status_code=400, detail=f'invalid option for "{definition.key}"')

            resolved[definition.key] = value
            continue

        if definition.type == "text":
            if not isinstance(value, str):
                raise HTTPException(status_code=400, detail=f'invalid value for "{definition.key}"')
            resolved[definition.key] = value
            continue

        if definition.type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise HTTPException(status_code=400, detail=f'invalid value for "{definition.key}"')

            _validate_numeric_range(definition, value)
            resolved[definition.key] = value
            continue

        if definition.type == "number":
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise HTTPException(status_code=400, detail=f'invalid value for "{definition.key}"')

            normalized_value = float(value)
            _validate_numeric_range(definition, normalized_value)
            resolved[definition.key] = normalized_value
            continue

        raise HTTPException(status_code=500, detail=f'unsupported setting type "{definition.type}"')

    return resolved


def _validate_numeric_range(definition: AdditionalSettingDefinitionData, value: int | float) -> None:
    if definition.min_value is not None and value < definition.min_value:
        raise HTTPException(status_code=400, detail=f'"{definition.key}" must be at least {definition.min_value}')
    if definition.max_value is not None and value > definition.max_value:
        raise HTTPException(status_code=400, detail=f'"{definition.key}" must be at most {definition.max_value}')

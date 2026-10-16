from fastapi import HTTPException

from app.models_registry import SEGMENTATION_MODELS, SegmentModelRequest, validate_additional_settings
from app.processes.base import ProcessStrategy
from app.schemas.process import SegmentFromPromptRequest


class SegmentFromPromptStrategy(ProcessStrategy[SegmentFromPromptRequest]):
    process_type = "segment_from_prompt"
    request_model = SegmentFromPromptRequest

    def execute(self, payload: SegmentFromPromptRequest) -> str:
        prompt = payload.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        input_image_url = payload.input_image_url.strip()
        if not input_image_url:
            raise HTTPException(status_code=400, detail="input image url is required")

        model_key = (payload.model_key or "sam3").strip()
        definition = SEGMENTATION_MODELS.get(model_key)
        if definition is None:
            raise HTTPException(status_code=400, detail="unsupported segmentation model")

        resolved_additional_settings = validate_additional_settings(
            definition.additional_settings,
            payload.additional_settings,
        )

        adapter = definition.adapter
        return adapter.run(
            SegmentModelRequest(
                input_image_url=input_image_url,
                prompt=prompt,
                project_id=payload.project_id,
                additional_settings=resolved_additional_settings,
            ),
        )

from fastapi import HTTPException

from app.models_registry import REMOVAL_MODELS, RemovalModelRequest, validate_additional_settings
from app.processes.base import ProcessStrategy
from app.schemas.process import RemoveWithMaskRequest


class RemoveWithMaskStrategy(ProcessStrategy[RemoveWithMaskRequest]):
    process_type = "remove_with_mask"
    request_model = RemoveWithMaskRequest

    def execute(self, payload: RemoveWithMaskRequest) -> str:
        input_image_url = payload.input_image_url.strip()
        if not input_image_url:
            raise HTTPException(status_code=400, detail="input image url is required")

        mask_image_url = payload.mask_image_url.strip()
        if not mask_image_url:
            raise HTTPException(status_code=400, detail="mask image url is required")

        model_key = (payload.model_key or "finegrain-eraser").strip()
        definition = REMOVAL_MODELS.get(model_key)
        if definition is None:
            raise HTTPException(status_code=400, detail="unsupported removal model")

        return definition.adapter.run(
            RemovalModelRequest(
                input_image_url=input_image_url,
                mask_image_url=mask_image_url,
                project_id=payload.project_id,
                additional_settings=validate_additional_settings(
                    definition.additional_settings,
                    payload.additional_settings,
                ),
            )
        )

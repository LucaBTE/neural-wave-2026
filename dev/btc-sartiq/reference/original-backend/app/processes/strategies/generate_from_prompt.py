from fastapi import HTTPException

from app.models_registry import GENERATION_MODELS, GenerationModelRequest, validate_additional_settings
from app.processes.base import ProcessStrategy
from app.schemas.process import GenerateFromPromptRequest


class GenerateFromPromptStrategy(ProcessStrategy[GenerateFromPromptRequest]):
    process_type = "generate_from_prompt"
    request_model = GenerateFromPromptRequest

    def execute(self, payload: GenerateFromPromptRequest) -> str:
        prompt = payload.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        input_image_url = payload.input_image_url.strip()
        if not input_image_url:
            raise HTTPException(status_code=400, detail="input image url is required")

        mask_image_url = payload.mask_image_url.strip()
        if not mask_image_url:
            raise HTTPException(status_code=400, detail="mask image url is required")

        model_key = (payload.model_key or "flux-fill-pro").strip()
        definition = GENERATION_MODELS.get(model_key)
        if definition is None:
            raise HTTPException(status_code=400, detail="unsupported generation model")

        adapter = definition.adapter
        return adapter.run(
            GenerationModelRequest(
                input_image_url=input_image_url,
                mask_image_url=mask_image_url,
                prompt=prompt,
                project_id=payload.project_id,
                additional_settings=validate_additional_settings(
                    definition.additional_settings,
                    payload.additional_settings,
                ),
            )
        )

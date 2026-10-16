from app.processes.registry import ProcessStrategyRegistry, build_process_strategy_registry
from app.processes.types import ProcessExecutionResult
from app.schemas.process import ProcessRunRequest


class ProcessExecutor:
    def __init__(self, registry: ProcessStrategyRegistry | None = None) -> None:
        self._registry = registry or build_process_strategy_registry()

    def execute(self, payload: ProcessRunRequest) -> ProcessExecutionResult:
        strategy = self._registry.get(payload)
        output_image_url = strategy.execute(payload)
        return ProcessExecutionResult(
            process_type=payload.process_type,
            output_image_url=output_image_url,
        )

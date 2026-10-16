import { useEffect, useMemo, useRef, useState } from "react";

import { uploadImageFromUrl } from "../api/images";
import {
  buildConvexHullPreview,
  createPipelineStep,
  finishPipeline,
  getPipeline,
  getPipelineSteps,
  getProcessCatalog,
  replacePipeline,
  runProcess,
  startPipeline,
} from "../api/processes";
import type {
  AdditionalSettingDefinition,
  ImageAsset,
  PipelineReplaceStepPayload,
  PipelineStep,
  ProcessCatalogItem,
  ProcessRunPayload,
  SegmentModel,
} from "../types/api";
import type { ConvexHullPreviewMode, LabCell } from "../types/laboratory";
import { getErrorMessage } from "../utils/errors";
import { getLaboratorySelectedImage } from "../utils/laboratorySelection";
import { toImageUrl } from "../utils/images";

const DEFAULT_FILL_PROMPT = "Fill the missing area naturally using the surrounding background.";
const DEFAULT_OUTPUT_CONVEX_HULL_MODE: ConvexHullPreviewMode = "medium";
export type PipelineSaveMode = "overwrite" | "save_as_new";
const NOTEBOOK_EXPLANATION_LIST = [
  "Standard pipeline:",
  "- Segmentation: write as prompt what you want to edit, ex. 'cat' 'bottle'",
  "- - you obtain the mask of that object. The mask will then be used in the future steps.",
  "- Remove: the mask is applied to the image and used to remove the object.",
  "- Fill: write as prompt what you want to add in the outlined area",
  "You can experiment quite freely with the workflow, but the first step should always be a segmentation!",
];

const FALLBACK_CATALOG: ProcessCatalogItem[] = [
  {
    process_type: "segment_from_prompt",
    title: "Segment",
    priority: 1,
    prompt_required: true,
    model_options: [{
      key: "sam3",
      label: "SAM 3.1",
      default: true,
      additional_settings: [
      ],
    }],
  },
  {
    process_type: "remove_with_mask",
    title: "Remove",
    priority: 2,
    prompt_required: false,
    model_options: [{
      key: "finegrain-eraser",
      label: "Finegrain Eraser",
      default: true,
      additional_settings: [
        {
          key: "mode",
          label: "Mode",
          type: "select",
          description: "Choose the removal quality mode used by Finegrain Eraser.",
          default_value: "standard",
          options: [
            { value: "express", label: "Express" },
            { value: "standard", label: "Standard" },
            { value: "premium", label: "Premium" },
          ],
        },
      ],
    }],
  },
  {
    process_type: "generate_from_prompt",
    title: "Fill",
    priority: 3,
    prompt_required: true,
    model_options: [{ key: "flux-fill-pro", label: "FLUX.1 [pro] Fill", default: true }],
  },
];

function getDefaultAdditionalSettings(
  modelOption: SegmentModel | undefined,
): Record<string, string | number | boolean> {
  return (modelOption?.additional_settings ?? []).reduce<Record<string, string | number | boolean>>(
    (acc, setting) => {
      if (setting.default_value !== undefined && setting.default_value !== null) {
        acc[setting.key] = setting.default_value;
      }
      return acc;
    },
    {},
  );
}

function sanitizeAdditionalSettings(
  settings: Record<string, string | number | boolean | null | undefined>,
): Record<string, string | number | boolean> {
  return Object.fromEntries(
    Object.entries(settings).filter(([, value]) => value !== "" && value !== null && value !== undefined),
  ) as Record<string, string | number | boolean>;
}

function getSelectedImageFromSession(): ImageAsset | null {
  return getLaboratorySelectedImage();
}

function createCell(def: ProcessCatalogItem): LabCell {
  const modelOptions = def.model_options ?? [];
  const defaultModel = (def.model_options ?? []).find((m) => m.default) ?? def.model_options?.[0];
  const defaultPrompt =
    def.process_type === "generate_from_prompt" ? DEFAULT_FILL_PROMPT : "";

  return {
    id: `${def.process_type}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    processType: def.process_type,
    title: def.title,
    priority: def.priority,
    promptRequired: def.prompt_required,
    modelOptions,
    prompt: defaultPrompt,
    modelKey: defaultModel?.key ?? "",
    additionalSettings: getDefaultAdditionalSettings(defaultModel),
    originalOutputUrl: "",
    outputConvexHullEnabled: false,
    outputConvexHullMode: DEFAULT_OUTPUT_CONVEX_HULL_MODE,
    outputPreviewLoading: false,
    status: "idle",
    outputUrl: "",
    error: "",
  };
}

function getEditableImageUrlBeforeCell(
  cellIndex: number,
  sourceCells: LabCell[],
  selectedImage: ImageAsset | null,
) {
  if (!selectedImage) {
    return "";
  }

  const cell = sourceCells[cellIndex];
  if (!cell) {
    return "";
  }

  for (let i = cellIndex - 1; i >= 0; i -= 1) {
    const candidate = sourceCells[i];
    if (candidate.processType === "segment_from_prompt") {
      continue;
    }

    const candidateOutputUrl = candidate.outputUrl.trim();
    return candidateOutputUrl ? toImageUrl(candidateOutputUrl) : "";
  }

  return toImageUrl(selectedImage.filePath);
}

function getSegmentationContext(cellIndex: number, sourceCells: LabCell[], selectedImage: ImageAsset | null) {
  for (let i = cellIndex - 1; i >= 0; i -= 1) {
    const candidate = sourceCells[i];
    if (candidate.processType !== "segment_from_prompt") {
      continue;
    }

    const maskOutputUrl = candidate.outputUrl.trim();
    if (!maskOutputUrl) {
      return null;
    }

    const inputImageUrl = getEditableImageUrlBeforeCell(i, sourceCells, selectedImage);
    if (!inputImageUrl) {
      return null;
    }

    return {
      inputImageUrl,
      maskImageUrl: toImageUrl(maskOutputUrl),
    };
  }

  return null;
}

function getEffectiveInputForCell(
  cellIndex: number,
  sourceCells: LabCell[],
  selectedImage: ImageAsset | null,
) {
  return getEditableImageUrlBeforeCell(cellIndex, sourceCells, selectedImage);
}

function createCellFromStep(def: ProcessCatalogItem, step: PipelineStep): LabCell {
  const baseCell = createCell(def);
  const additionalSettings = step.additional_settings_json ?? {};
  const prompt =
    def.process_type === "generate_from_prompt"
      ? step.prompt?.trim() || DEFAULT_FILL_PROMPT
      : step.prompt ?? "";

  return {
    ...baseCell,
    prompt,
    modelKey: step.model_key ?? baseCell.modelKey,
    additionalSettings: {
      ...baseCell.additionalSettings,
      ...additionalSettings,
    },
    originalOutputUrl: step.output_image_url ?? "",
    outputConvexHullEnabled: false,
    outputConvexHullMode: DEFAULT_OUTPUT_CONVEX_HULL_MODE,
    outputPreviewLoading: false,
    status: step.status === "done" ? "done" : "failed",
    outputUrl: step.output_image_url ?? "",
    error: step.error_message ?? "",
  };
}

function getPipelineNameFromImage(selectedImage: ImageAsset | null) {
  if (!selectedImage) {
    return "";
  }

  return selectedImage.fileName?.startsWith("Pipeline #") ? "" : selectedImage.fileName ?? "";
}

function syncCellWithDefinition(cell: LabCell, def: ProcessCatalogItem): LabCell {
  const modelOptions = def.model_options ?? [];
  const defaultModel = (def.model_options ?? []).find((model) => model.default) ?? def.model_options?.[0];
  const hasCurrentModel = modelOptions.some((model) => model.key === cell.modelKey);
  const nextModelKey = hasCurrentModel ? cell.modelKey : defaultModel?.key ?? "";
  const selectedModel = modelOptions.find((model) => model.key === nextModelKey);
  const allowedKeys = new Set((selectedModel?.additional_settings ?? []).map((setting) => setting.key));
  const filteredAdditionalSettings = Object.fromEntries(
    Object.entries(cell.additionalSettings).filter(([key]) => allowedKeys.has(key)),
  );

  return {
    ...cell,
    title: def.title,
    priority: def.priority,
    promptRequired: def.prompt_required,
    modelOptions,
    modelKey: nextModelKey,
    additionalSettings: hasCurrentModel
      ? { ...getDefaultAdditionalSettings(selectedModel), ...filteredAdditionalSettings }
      : getDefaultAdditionalSettings(selectedModel),
  };
}

export function getSelectedModelOption(cell: LabCell) {
  return cell.modelOptions.find((model) => model.key === cell.modelKey);
}

export function getMaskOverlayForCell(
  cellIndex: number,
  sourceCells: LabCell[],
  selectedImage: ImageAsset | null,
) {
  const cell = sourceCells[cellIndex];
  if (!cell || (cell.processType !== "generate_from_prompt" && cell.processType !== "remove_with_mask")) {
    return "";
  }

  const segmentationContext = getSegmentationContext(cellIndex, sourceCells, selectedImage);
  return segmentationContext?.maskImageUrl ?? "";
}

export function getVisibleAdditionalSettings(
  settingDefinitions: AdditionalSettingDefinition[],
  currentValues: Record<string, string | number | boolean>,
) {
  return settingDefinitions.filter((setting) => {
    if (!setting.depends_on_key) {
      return true;
    }

    return currentValues[setting.depends_on_key] === setting.depends_on_value;
  });
}

export function useLaboratoryNotebook() {
  const urlParams = useMemo(() => new URLSearchParams(window.location.search), []);
  const queryProjectId = urlParams.get("projectId");
  const queryImageId = urlParams.get("imageId");
  const queryPipelineId = urlParams.get("pipelineId");
  const [selectedImage, setSelectedImage] = useState<ImageAsset | null>(() => getSelectedImageFromSession());
  const [activePipelineId, setActivePipelineId] = useState<string | null>(() =>
    queryPipelineId ? queryPipelineId : null,
  );
  const [catalog, setCatalog] = useState<ProcessCatalogItem[]>(FALLBACK_CATALOG);
  const [cells, setCells] = useState<LabCell[]>([]);
  const [addProcessTypeByAnchor, setAddProcessTypeByAnchor] = useState<Record<string, string>>({});
  const [runningAll, setRunningAll] = useState(false);
  const [savingPipeline, setSavingPipeline] = useState(false);
  const [savingCellId, setSavingCellId] = useState("");
  const [saveMessageByCell, setSaveMessageByCell] = useState<Record<string, string>>({});
  const [saveErrorByCell, setSaveErrorByCell] = useState<Record<string, string>>({});
  const [saveMessage, setSaveMessage] = useState("");
  const [saveError, setSaveError] = useState("");
  const [loadingPipeline, setLoadingPipeline] = useState(false);
  const hasInitializedDefaultCell = useRef(false);

  useEffect(() => {
    let cancelled = false;

    async function loadCatalog() {
      try {
        const loaded = await getProcessCatalog();
        if (!cancelled && loaded.length > 0) {
          setCatalog(loaded);
        }
      } catch {
        // keep fallback catalog
      }
    }

    void loadCatalog();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!queryPipelineId || catalog.length === 0) {
      return;
    }

    let cancelled = false;
    const pipelineId = queryPipelineId;

    async function loadPipelineData() {
      setLoadingPipeline(true);
      try {
        const [pipeline, steps] = await Promise.all([getPipeline(pipelineId), getPipelineSteps(pipelineId)]);
        if (cancelled) {
          return;
        }

        setActivePipelineId(pipeline.id);
        setSelectedImage({
          id: pipeline.source_image_id,
          project_id: pipeline.project_id,
          fileName: pipeline.name?.trim() || `Pipeline #${pipeline.id} input`,
          filePath: pipeline.start_image_url,
          created_at: pipeline.created_at,
        });

        const loadedCells = [...steps]
          .sort((a, b) => a.step_index - b.step_index || a.id.localeCompare(b.id))
          .map((step) => {
            const definition = catalog.find((item) => item.process_type === step.process_type);
            if (!definition) return null;
            return createCellFromStep(definition, step);
          })
          .filter((cell): cell is LabCell => cell !== null);

        setCells(loadedCells);
        hasInitializedDefaultCell.current = true;
      } catch {
        // keep default laboratory behavior
      } finally {
        if (!cancelled) {
          setLoadingPipeline(false);
        }
      }
    }

    void loadPipelineData();
    return () => {
      cancelled = true;
    };
  }, [catalog, queryPipelineId]);

  useEffect(() => {
    if (!selectedImage || hasInitializedDefaultCell.current || cells.length > 0) {
      return;
    }

    const defaultProcess =
      catalog.find((processItem) => processItem.process_type === "segment_from_prompt") ?? catalog[0];

    if (!defaultProcess) {
      return;
    }

    setCells([createCell(defaultProcess)]);
    hasInitializedDefaultCell.current = true;
  }, [catalog, cells.length, selectedImage]);

  useEffect(() => {
    if (activePipelineId === null) {
      return;
    }

    const params = new URLSearchParams(window.location.search);
    if (params.get("pipelineId") === String(activePipelineId)) {
      return;
    }

    params.set("pipelineId", String(activePipelineId));
    window.history.replaceState({}, "", `/laboratory?${params.toString()}`);
  }, [activePipelineId]);

  useEffect(() => {
    if (catalog.length === 0) {
      return;
    }

    setCells((prev) => {
      let changed = false;

      const next = prev.map((cell) => {
        const matchingProcess = catalog.find((processItem) => processItem.process_type === cell.processType);
        if (!matchingProcess) {
          return cell;
        }

        const syncedCell = syncCellWithDefinition(cell, matchingProcess);
        if (
          syncedCell.title !== cell.title ||
          syncedCell.priority !== cell.priority ||
          syncedCell.promptRequired !== cell.promptRequired ||
          syncedCell.modelKey !== cell.modelKey ||
          syncedCell.modelOptions.length !== cell.modelOptions.length ||
          syncedCell.modelOptions.some(
            (modelOption, index) =>
              modelOption.key !== cell.modelOptions[index]?.key ||
              modelOption.label !== cell.modelOptions[index]?.label ||
              JSON.stringify(modelOption.additional_settings ?? []) !==
              JSON.stringify(cell.modelOptions[index]?.additional_settings ?? []),
          ) ||
          JSON.stringify(syncedCell.additionalSettings) !== JSON.stringify(cell.additionalSettings)
        ) {
          changed = true;
          return syncedCell;
        }

        return cell;
      });

      return changed ? next : prev;
    });
  }, [catalog]);

  const notebookExplanationList = NOTEBOOK_EXPLANATION_LIST;

  function getAddAnchorKey(afterIndex: number) {
    if (afterIndex < 0) return "start";
    return `after-${cells[afterIndex]?.id ?? afterIndex}`;
  }

  function getAvailableProcessesAfter(afterIndex: number, sourceCells: LabCell[] = cells) {
    const previousCell = afterIndex >= 0 ? sourceCells[afterIndex] : undefined;
    const previousPriority =
      previousCell?.processType === "remove_with_mask" || previousCell?.processType === "generate_from_prompt"
        ? 0
        : previousCell?.priority ?? 0;
    const nextPriority =
      afterIndex + 1 < sourceCells.length
        ? sourceCells[afterIndex + 1]?.priority ?? Number.MAX_SAFE_INTEGER
        : Number.MAX_SAFE_INTEGER;

    return catalog.filter(
      (processItem) => processItem.priority >= previousPriority && processItem.priority <= nextPriority,
    );
  }

  function getSelectedProcessTypeFor(afterIndex: number) {
    const available = getAvailableProcessesAfter(afterIndex);
    if (available.length === 0) return "";

    const anchorKey = getAddAnchorKey(afterIndex);
    const selected = addProcessTypeByAnchor[anchorKey];
    if (available.some((processItem) => processItem.process_type === selected)) {
      return selected;
    }

    const previousProcessType = afterIndex >= 0 ? cells[afterIndex]?.processType : undefined;
    const preferredNextProcessType =
      previousProcessType === "segment_from_prompt"
        ? "remove_with_mask"
        : previousProcessType === "remove_with_mask"
          ? "generate_from_prompt"
          : previousProcessType === "generate_from_prompt"
            ? "segment_from_prompt"
            : undefined;

    if (preferredNextProcessType) {
      const preferredProcess = available.find((processItem) => processItem.process_type === preferredNextProcessType);
      if (preferredProcess) {
        return preferredProcess.process_type;
      }
    }

    return available[0].process_type;
  }

  function canAddProcessAfter(afterIndex: number, sourceCells: LabCell[] = cells) {
    if (afterIndex < 0) {
      return Boolean(selectedImage);
    }

    const previousCell = sourceCells[afterIndex];
    return previousCell?.status === "done";
  }

  function setSelectedProcessTypeFor(afterIndex: number, processType: string) {
    const anchorKey = getAddAnchorKey(afterIndex);
    setAddProcessTypeByAnchor((prev) => ({ ...prev, [anchorKey]: processType }));
  }

  function addCell(afterIndex: number) {
    const selectedProcessType = getSelectedProcessTypeFor(afterIndex);
    const selected = catalog.find((p) => p.process_type === selectedProcessType);
    if (!selected) return;

    setCells((prev) => {
      const next = [...prev];
      const insertAt = afterIndex + 1;
      next.splice(insertAt, 0, createCell(selected));

      for (let i = insertAt; i < next.length; i += 1) {
        if (i === insertAt) continue;
        next[i] = {
          ...next[i],
          status: "idle",
          outputUrl: "",
          originalOutputUrl: "",
          outputConvexHullEnabled: false,
          outputPreviewLoading: false,
          error: "",
        };
      }

      return next;
    });
    setSaveMessage("");
    setSaveError("");
  }

  function updateCell(cellIndex: number, patch: Partial<LabCell>) {
    setCells((prev) => {
      const next = [...prev];
      next[cellIndex] = { ...next[cellIndex], ...patch };
      return next;
    });
  }

  function updateModelForCell(cellIndex: number, modelKey: string) {
    setCells((prev) => {
      const next = [...prev];
      const cell = next[cellIndex];
      if (!cell) return prev;

      const selectedModel = cell.modelOptions.find((model) => model.key === modelKey);
      next[cellIndex] = {
        ...cell,
        modelKey,
        additionalSettings: getDefaultAdditionalSettings(selectedModel),
      };
      return next;
    });
  }

  function updateAdditionalSetting(cellIndex: number, settingKey: string, value: string | number | boolean) {
    setCells((prev) => {
      const next = [...prev];
      const cell = next[cellIndex];
      if (!cell) return prev;

      next[cellIndex] = {
        ...cell,
        additionalSettings: {
          ...cell.additionalSettings,
          [settingKey]: value,
        },
      };
      return next;
    });
  }

  function getInputForCell(cellIndex: number, sourceCells: LabCell[]) {
    return getEffectiveInputForCell(cellIndex, sourceCells, selectedImage);
  }

  function getMaskOverlayUrlForCell(cellIndex: number, sourceCells: LabCell[]) {
    return getMaskOverlayForCell(cellIndex, sourceCells, selectedImage);
  }

  function resetFromCell(cellIndex: number) {
    setCells((prev) => prev.slice(0, cellIndex + 1));
    setSaveMessage("");
    setSaveError("");
  }

  function removeCell(cellIndex: number) {
    if (cellIndex === 0) {
      return;
    }

    setCells((prev) => {
      const next = prev.filter((_, index) => index !== cellIndex);
      for (let i = cellIndex; i < next.length; i += 1) {
        next[i] = {
          ...next[i],
          status: "idle",
          outputUrl: "",
          originalOutputUrl: "",
          outputConvexHullEnabled: false,
          outputPreviewLoading: false,
          error: "",
        };
      }
      return next;
    });
    setSaveMessage("");
    setSaveError("");
  }

  async function setSegmentOutputConvexHull(
    cellIndex: number,
    enabled: boolean,
    mode?: ConvexHullPreviewMode,
  ) {
    const currentCells = [...cells];
    const cell = currentCells[cellIndex];
    if (!cell || cell.processType !== "segment_from_prompt") {
      return;
    }

    const baseOutputUrl = cell.originalOutputUrl || cell.outputUrl;
    if (!baseOutputUrl) {
      return;
    }

    const nextMode = mode ?? cell.outputConvexHullMode ?? DEFAULT_OUTPUT_CONVEX_HULL_MODE;

    if (!enabled) {
      updateCell(cellIndex, {
        outputConvexHullEnabled: false,
        outputPreviewLoading: false,
        outputUrl: baseOutputUrl,
        error: "",
      });
      return;
    }

    updateCell(cellIndex, {
      outputPreviewLoading: true,
      error: "",
      outputConvexHullMode: nextMode,
    });

    try {
      const response = await buildConvexHullPreview({
        mask_image_url: toImageUrl(baseOutputUrl),
        mode: nextMode,
      });
      updateCell(cellIndex, {
        outputConvexHullEnabled: true,
        outputConvexHullMode: nextMode,
        outputPreviewLoading: false,
        outputUrl: response.output_image_url,
        error: "",
      });
    } catch {
      updateCell(cellIndex, {
        outputConvexHullEnabled: false,
        outputPreviewLoading: false,
        outputUrl: baseOutputUrl,
        error: "Unable to build convex hull preview from the current mask",
      });
    }
  }

  async function setSegmentOutputConvexHullMode(cellIndex: number, mode: ConvexHullPreviewMode) {
    setCells((prev) => {
      const next = [...prev];
      const cell = next[cellIndex];
      if (!cell) return prev;
      next[cellIndex] = {
        ...cell,
        outputConvexHullMode: mode,
      };
      return next;
    });

    const currentCell = cells[cellIndex];
    if (!currentCell) {
      return;
    }

    await setSegmentOutputConvexHull(cellIndex, true, mode);
  }

  function buildPayload(
    cellIndex: number,
    cell: LabCell,
    sourceCells: LabCell[],
    inputImageUrl: string,
  ): ProcessRunPayload {
    const basePayload: ProcessRunPayload = {
      process_type: cell.processType,
      priority: cell.priority,
      pipeline_id: activePipelineId ?? undefined,
      step_index: cellIndex + 1,
      project_id: queryProjectId ?? selectedImage?.project_id,
      image_id: queryImageId ?? selectedImage?.id,
    };

    if (cell.processType === "segment_from_prompt") {
      const additionalSettings = sanitizeAdditionalSettings(cell.additionalSettings);

      return {
        ...basePayload,
        prompt: cell.prompt,
        input_image_url: inputImageUrl,
        model_key: cell.modelKey || undefined,
        additional_settings: Object.keys(additionalSettings).length > 0 ? additionalSettings : undefined,
      };
    }

    if (cell.processType === "remove_with_mask") {
      const additionalSettings = sanitizeAdditionalSettings(cell.additionalSettings);
      const segmentationContext = getSegmentationContext(cellIndex, sourceCells, selectedImage);

      return {
        ...basePayload,
        input_image_url: inputImageUrl,
        mask_image_url: segmentationContext?.maskImageUrl,
        model_key: cell.modelKey || undefined,
        additional_settings: Object.keys(additionalSettings).length > 0 ? additionalSettings : undefined,
      };
    }

    if (cell.processType === "generate_from_prompt") {
      const additionalSettings = sanitizeAdditionalSettings(cell.additionalSettings);

      const segmentationContext = getSegmentationContext(cellIndex, sourceCells, selectedImage);
      return {
        ...basePayload,
        prompt: cell.prompt,
        model_key: cell.modelKey || undefined,
        additional_settings: Object.keys(additionalSettings).length > 0 ? additionalSettings : undefined,
        input_image_url: inputImageUrl,
        mask_image_url: segmentationContext?.maskImageUrl,
      };
    }

    return {
      ...basePayload,
      prompt: cell.prompt,
      model_key: cell.modelKey || undefined,
    };
  }

  async function runCell(cellIndex: number) {
    if (!selectedImage) return null;

    const currentCells = [...cells];
    const cell = currentCells[cellIndex];
    if (!cell) return null;

    const inputImageUrl = getInputForCell(cellIndex, currentCells);

    if (!inputImageUrl) {
      updateCell(cellIndex, { status: "failed", error: "Previous cell output missing" });
      return null;
    }

    if (cell.processType === "remove_with_mask") {
      const segmentationContext = getSegmentationContext(cellIndex, currentCells, selectedImage);
      if (!segmentationContext?.maskImageUrl) {
        updateCell(cellIndex, {
          status: "failed",
          error: "Removal requires a mask from a previous segmentation cell",
        });
        return null;
      }
    }

    if (cell.processType === "generate_from_prompt") {
      const segmentationContext = getSegmentationContext(cellIndex, currentCells, selectedImage);
      if (!segmentationContext?.maskImageUrl) {
        updateCell(cellIndex, {
          status: "failed",
          error: "Generation requires a mask from a previous segmentation cell",
        });
        return null;
      }
    }

    if (cell.promptRequired && !cell.prompt.trim()) {
      const message = "Prompt is required for this process.";
      window.alert(message);
      updateCell(cellIndex, { status: "failed", error: message });
      return null;
    }

    updateCell(cellIndex, { status: "running", error: "" });
    setSaveMessage("");
    setSaveError("");

    try {
      const payload = buildPayload(cellIndex, cell, currentCells, inputImageUrl);
      const response = await runProcess(payload);
      updateCell(cellIndex, {
        status: "done",
        outputUrl: response.output_image_url,
        originalOutputUrl: response.output_image_url,
        outputConvexHullEnabled: false,
        outputConvexHullMode: DEFAULT_OUTPUT_CONVEX_HULL_MODE,
        outputPreviewLoading: false,
        error: "",
      });
      return response.output_image_url;
    } catch (error) {
      updateCell(cellIndex, {
        status: "failed",
        error: error instanceof Error ? error.message : "Unexpected error",
      });
      return null;
    }
  }

  async function runAllCells() {
    if (!selectedImage || cells.length === 0) return;

    setRunningAll(true);
    setSaveMessage("");
    setSaveError("");

    let allSuccessful = true;
    let lastOutputUrl = "";

    for (let i = 0; i < cells.length; i += 1) {
      const outputUrl = await runCell(i);
      if (!outputUrl) {
        allSuccessful = false;
        break;
      }
      lastOutputUrl = outputUrl;
    }

    if (activePipelineId) {
      try {
        await finishPipeline(activePipelineId, {
          status: allSuccessful ? "done" : "failed",
          final_image_url: allSuccessful ? lastOutputUrl : undefined,
        });
      } catch {
        // no-op
      }
    }

    setRunningAll(false);
  }

  function buildPipelineStepsSnapshot(): PipelineReplaceStepPayload[] {
    if (!selectedImage) {
      return [];
    }

    return cells.flatMap((cell, index) => {
      if (cell.status !== "done" && cell.status !== "failed") {
        return [];
      }

      const inputImageUrl = getInputForCell(index, cells);
      const segmentationContext = getSegmentationContext(index, cells, selectedImage);

      return [{
        step_index: index + 1,
        process_type: cell.processType,
        priority: cell.priority,
        model_key: cell.modelKey || undefined,
        prompt: cell.prompt || undefined,
        additional_settings_json: sanitizeAdditionalSettings(cell.additionalSettings),
        input_image_url: inputImageUrl,
        mask_image_url:
          cell.processType === "remove_with_mask" || cell.processType === "generate_from_prompt"
            ? segmentationContext?.maskImageUrl
            : undefined,
        output_image_url: cell.outputUrl || undefined,
        status: cell.status === "done" ? "done" : "failed",
        error_message: cell.error || undefined,
      }];
    });
  }

  async function savePipeline(mode: PipelineSaveMode, name: string) {
    if (!selectedImage) {
      setSaveError("Select an image first");
      return false;
    }

    const latestOutputUrl = [...cells].reverse().find((cell) => cell.outputUrl && cell.status === "done")?.outputUrl;

    if (!latestOutputUrl) {
      setSaveError("Run at least one cell before saving");
      return false;
    }

    const nextName = name.trim() || undefined;
    const steps = buildPipelineStepsSnapshot();

    try {
      setSavingPipeline(true);

      if (mode === "overwrite" && activePipelineId) {
        const updated = await replacePipeline(activePipelineId, {
          name: nextName,
          status: "done",
          final_image_url: latestOutputUrl,
          steps,
        });

        setSelectedImage((prev) =>
          prev
            ? {
              ...prev,
              fileName: updated.name?.trim() || prev.fileName,
            }
            : prev,
        );
        setSaveError("");
        setSaveMessage("Pipeline saved");
        return true;
      }

      let pipelineId = activePipelineId;
      if (!pipelineId || mode === "save_as_new") {
        const projectId = queryProjectId ?? selectedImage.project_id;
        const imageId = queryImageId ?? selectedImage.id;
        if (!projectId || !imageId) {
          setSaveError("Project or image id missing");
          return false;
        }

        const created = await startPipeline({
          project_id: projectId,
          source_image_id: imageId,
          start_image_url: toImageUrl(selectedImage.filePath),
          name: nextName,
        });
        pipelineId = created.id;
        setActivePipelineId(created.id);
      }

      if (!pipelineId) {
        setSaveError("Could not create the pipeline");
        return false;
      }

      for (const step of steps) {
        await createPipelineStep(pipelineId, step);
      }

      const updated = await finishPipeline(pipelineId, {
        status: "done",
        final_image_url: latestOutputUrl,
      });

      setSelectedImage((prev) =>
        prev
          ? {
            ...prev,
            fileName: updated.name?.trim() || prev.fileName,
          }
          : prev,
      );
      setSaveError("");
      setSaveMessage("Pipeline saved");
      return true;
    } catch (error) {
      setSaveMessage("");
      setSaveError(getErrorMessage(error));
      return false;
    } finally {
      setSavingPipeline(false);
    }
  }

  async function saveCellOutputToProject(cell: LabCell) {
    if (!cell.outputUrl || !selectedImage) return;

    const targetProjectId = queryProjectId ?? selectedImage.project_id;
    if (!targetProjectId) {
      setSaveErrorByCell((prev) => ({ ...prev, [cell.id]: "Project id missing" }));
      return;
    }

    setSavingCellId(cell.id);
    setSaveMessageByCell((prev) => ({ ...prev, [cell.id]: "" }));
    setSaveErrorByCell((prev) => ({ ...prev, [cell.id]: "" }));

    try {
      await uploadImageFromUrl(targetProjectId, toImageUrl(cell.outputUrl), `${cell.processType}-${Date.now()}`);

      if (activePipelineId) {
        try {
          await finishPipeline(activePipelineId, {
            status: "done",
            final_image_url: cell.outputUrl,
          });
        } catch {
          // no-op
        }
      }

      const successMessage = `Saved to project`;
      setSaveMessageByCell((prev) => ({ ...prev, [cell.id]: successMessage }));
      setSaveErrorByCell((prev) => ({ ...prev, [cell.id]: "" }));
    } catch (error) {
      const errorMessage = getErrorMessage(error);
      setSaveErrorByCell((prev) => ({ ...prev, [cell.id]: errorMessage }));
    } finally {
      setSavingCellId("");
    }
  }

  return {
    queryProjectId,
    queryImageId,
    selectedImage,
    activePipelineId,
    catalog,
    cells,
    runningAll,
    savingPipeline,
    savingCellId,
    saveMessageByCell,
    saveErrorByCell,
    saveMessage,
    saveError,
    loadingPipeline,
    currentPipelineName: getPipelineNameFromImage(selectedImage),
    notebookExplanationList,
    getAvailableProcessesAfter,
    getSelectedProcessTypeFor,
    canAddProcessAfter,
    setSelectedProcessTypeFor,
    addCell,
    updateCell,
    updateModelForCell,
    updateAdditionalSetting,
    getInputForCell,
    getMaskOverlayUrlForCell,
    resetFromCell,
    removeCell,
    runCell,
    runAllCells,
    savePipeline,
    saveCellOutputToProject,
    setSegmentOutputConvexHull,
    setSegmentOutputConvexHullMode,
  };
}

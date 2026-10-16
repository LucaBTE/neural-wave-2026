import { request } from "./client";
import type {
  ConvexHullPreviewPayload,
  ConvexHullPreviewResponse,
  Pipeline,
  PipelineFinishPayload,
  PipelineReplacePayload,
  PipelineStartPayload,
  PipelineStep,
  ProcessRunPayload,
  ProcessRunResponse,
  ProcessCatalogItem,
  SegmentModel,
} from "../types/api";

export function runProcess(payload: ProcessRunPayload) {
  return request<ProcessRunResponse>("/processes/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function buildConvexHullPreview(payload: ConvexHullPreviewPayload) {
  return request<ConvexHullPreviewResponse>("/processes/convex-hull-preview", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getSegmentModels() {
  return request<SegmentModel[]>("/processes/segment-models");
}

export function getRemovalModels() {
  return request<SegmentModel[]>("/processes/remove-models");
}

export function getProcessCatalog() {
  return request<ProcessCatalogItem[]>("/processes/catalog");
}

export function listPipelines() {
  return request<Pipeline[]>("/laboratory-pipelines");
}

export function getPipeline(pipelineId: string) {
  return request<Pipeline>(`/laboratory-pipelines/${pipelineId}`);
}

export function startPipeline(payload: PipelineStartPayload) {
  return request<Pipeline>("/laboratory-pipelines/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function finishPipeline(pipelineId: string, payload: PipelineFinishPayload) {
  return request<Pipeline>(`/laboratory-pipelines/${pipelineId}/finish`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getPipelineSteps(pipelineId: string) {
  return request<PipelineStep[]>(`/laboratory-pipelines/${pipelineId}/steps`);
}

export function createPipelineStep(
  pipelineId: string,
  payload: {
    step_index: number;
    process_type: string;
    priority: number;
    model_key?: string;
    prompt?: string;
    additional_settings_json?: Record<string, string | number | boolean>;
    input_image_url: string;
    mask_image_url?: string;
    output_image_url?: string;
    status: string;
    error_message?: string;
  },
) {
  return request<PipelineStep>(`/laboratory-pipelines/${pipelineId}/steps`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function renamePipeline(pipelineId: string, name: string) {
  return request<Pipeline>(`/laboratory-pipelines/${pipelineId}/name`, {
    method: "PATCH",
    body: JSON.stringify({ name }),
  });
}

export function replacePipeline(pipelineId: string | number, payload: PipelineReplacePayload) {
  return request<Pipeline>(`/laboratory-pipelines/${pipelineId}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function deletePipeline(pipelineId: string | number) {
  return request<void>(`/laboratory-pipelines/${pipelineId}`, {
    method: "DELETE",
  });
}

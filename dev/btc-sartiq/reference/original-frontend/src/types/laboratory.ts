import type { ProcessStatus } from "./process";
import type { SegmentModel } from "./api";

export type ConvexHullPreviewMode = "simple" | "medium" | "rectangle";

export interface LabCell {
  id: string;
  processType: string;
  title: string;
  priority: number;
  promptRequired: boolean;
  modelOptions: SegmentModel[];
  prompt: string;
  modelKey: string;
  additionalSettings: Record<string, string | number | boolean>;
  originalOutputUrl: string;
  outputConvexHullEnabled: boolean;
  outputConvexHullMode: ConvexHullPreviewMode;
  outputPreviewLoading: boolean;
  status: ProcessStatus;
  outputUrl: string;
  error: string;
}

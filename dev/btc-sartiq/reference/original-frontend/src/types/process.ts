export type ProcessKind = "segment_from_prompt" | "remove_with_mask" | "generate_from_prompt";

export type ProcessStatus = "idle" | "running" | "done" | "failed";

export interface ProcessModelOption {
  key: string;
  label: string;
}

export interface ProcessStep {
  id: string;
  kind: ProcessKind;
  title: string;
  promptPlaceholder: string;
  promptRequired: boolean;
  modelOptions?: ProcessModelOption[];
}

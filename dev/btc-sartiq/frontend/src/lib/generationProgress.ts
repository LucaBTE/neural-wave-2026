const finalStages: Record<string, [number, number, number]> = {
  generating: [90, 95, 30], pressing: [94, 97, 18],
  finishing: [96, 98.5, 8], checking_quality: [98, 99, 5],
  complete: [99, 99, 3], review_required: [99, 99, 3],
};

/** A time-based estimate; completion is handled separately after image decode. */
export function estimateGenerationProgress(elapsed: number, status: string, stageAge: number): number {
  if (elapsed <= 40) return Math.max(0, elapsed) * 90 / 40;
  const [start, ceiling, duration] = finalStages[status] || [90, 93, 45];
  const age = Math.max(0, Math.min(stageAge, elapsed - 40));
  return Math.min(99, start + (ceiling - start) * (1 - Math.exp(-age / duration)));
}

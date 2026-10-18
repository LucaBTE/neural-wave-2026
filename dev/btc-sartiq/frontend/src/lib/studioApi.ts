type StudioResponse = {job_id?: unknown; detail?: unknown};

export async function startStudioJob(photos: File[]): Promise<string> {
  const body = new FormData();
  body.append('file', photos[0]);
  photos.slice(1).forEach(photo => body.append('additional_files', photo));
  let response: Response;
  try {
    response = await fetch('/api/studio', {method: 'POST', body});
  } catch {
    throw new Error('Cannot reach the generation service. Check your connection and that the backend is running.');
  }
  const text = await response.text();
  let data: StudioResponse | undefined;
  try {data = JSON.parse(text);} catch { /* Proxies may return HTML or an empty error response. */ }
  if (!response.ok) {
    const detail = typeof data?.detail === 'string' ? data.detail : undefined;
    throw new Error(detail || (response.status >= 500
      ? `Generation service unavailable (HTTP ${response.status}). Check that the backend is running on port 8000.`
      : `Unable to start generation (HTTP ${response.status}). Please try again.`));
  }
  if (typeof data?.job_id !== 'string' || !/^[a-zA-Z0-9_-]+$/.test(data.job_id)) {
    throw new Error('The generation service returned an invalid response. Check the /api proxy and backend connection.');
  }
  return data.job_id;
}

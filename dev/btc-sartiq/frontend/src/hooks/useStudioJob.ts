import {useCallback, useEffect, useRef, useState} from 'react';
import {startStudioJob} from '../lib/studioApi';

export type StepImage = {id: string; label: string; url: string};
export type Job = {
  job_id: string;
  status: string;
  result_url?: string;
  error?: string;
  current_image?: StepImage;
  images?: StepImage[];
};

const terminalStates = new Set(['complete', 'review_required', 'error']);

export function useStudioJob() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const requestLock = useRef(false);

  const busy = submitting || Boolean(job && !terminalStates.has(job.status));

  useEffect(() => {
    if (!file) {
      setPreview('');
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  useEffect(() => {
    if (!job || terminalStates.has(job.status)) return;
    const events = new EventSource(`/api/jobs/${job.job_id}/events`);
    events.onmessage = event => {
      try {
        const next: Job = JSON.parse(event.data);
        setJob(next);
        setError(next.status === 'error' ? next.error || 'We could not finish this image.' : '');
        if (terminalStates.has(next.status)) events.close();
      } catch {
        setError('We could not read the latest update. Retrying…');
      }
    };
    events.onerror = () => {
      if (!terminalStates.has(job.status)) setError('Connection interrupted. Trying to reconnect…');
    };
    return () => events.close();
  }, [job?.job_id]);

  const selectFile = useCallback((next?: File) => {
    if (!next || busy) return false;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(next.type) || next.size > 12_000_000) {
      setError('Choose a JPG, PNG, or WebP photo up to 12 MB.');
      return false;
    }
    setFile(next);
    setJob(null);
    setError('');
    return true;
  }, [busy]);

  const generate = useCallback(async (selectedFile = file) => {
    if (!selectedFile || busy || requestLock.current) return false;
    requestLock.current = true;
    setSubmitting(true);
    setJob(null);
    setError('');
    try {
      const jobId = await startStudioJob([selectedFile]);
      setJob({job_id: jobId, status: 'queued'});
      return true;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      return false;
    } finally {
      setSubmitting(false);
      requestLock.current = false;
    }
  }, [busy, file]);

  const reset = useCallback(() => {
    if (busy) return;
    setFile(null);
    setJob(null);
    setError('');
  }, [busy]);

  return {file, preview, job, error, busy, submitting, selectFile, generate, reset, clearError: () => setError('')};
}

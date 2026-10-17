import {useEffect, useRef, useState} from 'react';
import {Upload, ArrowUpRight, ArrowRight, ChevronDown, Loader2, Image as ImageIcon} from 'lucide-react';

type StepImage = {id: string; label: string; url: string};
type TraceStep = {id: string; label: string; model: string | null; libraries: Record<string, string>; python: string;
  status: string; seconds: number | null; details: {cache_hit?: boolean; resolution?: string; model_invoked?: boolean}};
type Job = {job_id: string; status: string; result_url?: string; error?: string; current_image?: StepImage; images?: StepImage[]; execution_trace?: TraceStep[]};
const phases = [
  {status: 'preparing', label: 'Preparing'},
  {status: 'balancing_color', label: 'Balancing colors'},
  {status: 'describing', label: 'Analyzing garment'},
  {status: 'generating', label: 'Creating photo'},
  {status: 'pressing', label: 'Removing wrinkles'},
  {status: 'finishing', label: 'Refining'},
  {status: 'checking_quality', label: 'Final check'},
];
const finished = ['complete', 'review_required', 'error'];

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const requestLock = useRef(false);
  const busy = submitting || !!(job && !finished.includes(job.status));
  const currentImage = job?.result_url ? {id: 'final', label: 'Final output', url: job.result_url} : job?.current_image;
  const canViewHistory = !!job && finished.includes(job.status) && !!job.images?.length;
  const currentStep = job?.execution_trace?.[job.execution_trace.length - 1];

  const phaseIndex = phases.findIndex(phase => phase.status === job?.status);
  const progressLabel = submitting ? 'Uploading' : job?.status === 'queued' ? 'Queued' : phases[phaseIndex]?.label || 'Processing';

  useEffect(() => {
    if (!busy || startedAt === null) return;
    const tick = () => setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    tick();
    const timer = window.setInterval(tick, 1000);
    return () => window.clearInterval(timer);
  }, [busy, startedAt]);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!job || finished.includes(job.status)) return;
    const events = new EventSource('/api/jobs/' + job.job_id + '/events');
    events.onmessage = (event) => {
      try {
        const data: Job = JSON.parse(event.data);
        setJob(data);
        setError(data.status === 'error' ? data.error || 'Processing failed' : '');
        if (finished.includes(data.status)) events.close();
      } catch {
        setError('Unable to read progress updates.');
      }
    };
    events.onerror = () => setError('Connection lost. Reconnecting…');
    return () => events.close();
  }, [job?.job_id]);

  const select = (next?: File) => {
    if (!next || busy) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(next.type) || next.size > 12_000_000) {
      setError('Choose a JPG, PNG or WebP photo up to 12 MB.'); return;
    }
    setFile(next); setJob(null); setError(''); setShowHistory(false);
  };
  const generate = async () => {
    if (!file || busy || requestLock.current) return;
    requestLock.current = true;
    setSubmitting(true); setJob(null); setError(''); setStartedAt(Date.now()); setElapsed(0); setShowHistory(false);
    try {
      const body = new FormData(); body.append('file', file);
      const response = await fetch('/api/studio', {method: 'POST', body});
      const data = await response.json();
      if (!response.ok) throw Error(data.detail || 'Unable to start generation');
      setJob({job_id: data.job_id, status: 'queued'});
    } catch (e) {setError(e instanceof Error ? e.message : String(e));}
    finally {setSubmitting(false); requestLock.current = false;}
  };

  return <div className="app">
    <header><span className="brand">sartiq<span className="brand-dot">.</span></span><span className="header-label">Studio</span></header>
    <main>
      <h1>Catalogue photography.</h1>
      <input ref={input} type="file" accept="image/png,image/jpeg,image/webp" hidden disabled={busy} onChange={e => {select(e.target.files?.[0]); e.target.value = '';}}/>
      <div className="comparison">
        <section aria-label="Original photo">
          <div className="caption"><span>Original</span>{file && <span className="filename" title={file.name}>{file.name}</span>}</div>
          <button className={'photo upload ' + (dragging ? 'dragging' : '')} disabled={busy} aria-label={preview ? 'Change photo' : 'Upload a photo'}
            onClick={() => input.current?.click()} onDragOver={e => {e.preventDefault(); if (!busy) setDragging(true);}}
            onDragLeave={() => setDragging(false)} onDrop={e => {e.preventDefault(); setDragging(false); select(e.dataTransfer.files[0]);}}>
            {preview ? <><img src={preview} alt="Original garment"/><span className="change">Change photo</span></> :
              <div className="empty"><Upload size={24} strokeWidth={1.5}/><strong>Upload a photo</strong><span>or drop it here</span><small>JPG, PNG, WebP · max 12 MB</small></div>}
          </button>
        </section>
        <section aria-label="Current image" aria-busy={busy}>
          <div className="caption"><span>{busy ? 'Live preview' : 'Result'}</span>{job?.result_url && <span className="ready">Ready</span>}</div>
          <div className={'photo result ' + (currentImage?.id === 'cutout' ? 'transparent ' : '') + (busy ? 'processing' : '')}>
            {currentImage ? <><img src={currentImage.url} alt={currentImage.label}/><span className="live-caption">{busy && <Loader2 className="spinner" size={13}/>} {currentImage.label}</span></> :
              <div className="empty">{busy ? <><Loader2 className="spinner" size={28} strokeWidth={1.5}/><strong>Creating your studio shot</strong></> : <><ImageIcon size={26} strokeWidth={1.3}/><span>Your photo, on white.</span></>}</div>}
          </div>
        </section>
      </div>
      {busy && <div className="progress-panel">
        <div className="progress-heading"><span role="status">{progressLabel}<span className="phase-count">{phaseIndex >= 0 ? ` · ${phaseIndex + 1}/${phases.length}` : ''}</span></span>
          <span className="elapsed" aria-label="Elapsed time">{Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, '0')}</span></div>
        <div className="progress-track" role="progressbar" aria-label="Completed stages" aria-valuemin={0} aria-valuemax={phases.length} aria-valuenow={Math.max(0, phaseIndex)} aria-valuetext={progressLabel}>
          {phases.map((phase, i) => <span key={phase.status} className={i < phaseIndex ? 'done' : i === phaseIndex ? 'current' : ''}/>)}</div>
        <p className="wait-note">This may take a few minutes.</p>
      </div>}
      {error && <p className="error" role="alert">{error}</p>}
      {!!job?.execution_trace?.length && <details className="execution-details">
        <summary>Execution details{busy && currentStep && <span>{currentStep.model || Object.keys(currentStep.libraries).filter(name => name !== 'LangGraph').join(' · ')}</span>}</summary>
        <div className="execution-list">
          {job.execution_trace.map(step => <article key={step.id} className="execution-step">
            <div className="execution-heading"><strong>{step.label}</strong><span className={'execution-status ' + step.status}>
              {step.status.replace(/_/g, ' ')}{step.seconds !== null && ` · ${step.seconds.toFixed(2)}s`}
            </span></div>
            <div className="execution-model">{step.model || 'Local processing'}{step.details.resolution && ` · ${step.details.resolution}`}</div>
            <p>{Object.entries(step.libraries).map(([name, version]) => `${name} ${version}`).join(' · ')} · Python {step.python}</p>
            {step.details.cache_hit && <small>Cached description · no model request</small>}
          </article>)}
        </div>
      </details>}
      <div className="actions">
        <button className="primary" onClick={generate} disabled={!file || busy}>{busy ? 'Processing…' : job?.result_url ? 'Regenerate' : 'Generate photo'}{!busy && <ArrowRight size={17}/>}</button>
        {job?.result_url && <a className="open-result" href={job.result_url} target="_blank" rel="noreferrer">Open photo<ArrowUpRight size={17}/></a>}
        {canViewHistory && <button className="history-toggle" aria-expanded={showHistory} aria-controls="image-history" onClick={() => setShowHistory(value => !value)}>
          {showHistory ? 'Hide steps' : job?.status === 'error' ? 'View available steps' : 'View all steps'}<ChevronDown size={16} className={showHistory ? 'expanded' : ''}/>
        </button>}
      </div>
      {job?.result_url && <p className="review-note">{job.status === 'review_required' ? 'Check colors, prints and details before publishing.' : 'Check details before publishing.'}</p>}
      {canViewHistory && showHistory && <section id="image-history" className="image-history" aria-label="Intermediate images">
        {job?.images?.map((image, index) => <figure key={image.id}>
          <a className={'history-image ' + (image.id === 'cutout' ? 'transparent' : '')} href={image.url} target="_blank" rel="noreferrer" aria-label={'Open ' + image.label}>
            <img src={image.url} alt={image.label} loading="lazy"/>
          </a>
          <figcaption><span>{String(index + 1).padStart(2, '0')}</span>{image.label}</figcaption>
        </figure>)}
      </section>}
    </main>
  </div>;
}

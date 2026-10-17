import {useEffect, useRef, useState} from 'react';
import {Upload, ArrowUpRight, ArrowRight, ChevronDown, Loader2, Image as ImageIcon, Plus, X} from 'lucide-react';

type StepImage = {id: string; label: string; url: string};
type TraceStep = {id: string; label: string; model: string | null; libraries: Record<string, string>; python: string;
  status: string; seconds: number | null; details: {cache_hit?: boolean; resolution?: string; model_invoked?: boolean; view_count?: number; input_image_count?: number}};
type Job = {job_id: string; status: string; result_url?: string; error?: string; current_image?: StepImage; images?: StepImage[]; execution_trace?: TraceStep[];
  classification?: {review_required?: boolean}};
const phases = [
  {status: 'classifying', label: 'Identifying product'},
  {status: 'locating', label: 'Locating product'},
  {status: 'preparing', label: 'Preparing'},
  {status: 'balancing_color', label: 'Balancing colors'},
  {status: 'describing', label: 'Analyzing product'},
  {status: 'generating', label: 'Creating photo'},
  {status: 'pressing', label: 'Removing wrinkles'},
  {status: 'finishing', label: 'Refining'},
  {status: 'checking_quality', label: 'Final check'},
];
const finished = ['complete', 'review_required', 'error'];

export default function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const file = files[0];
  const preview = previews[0];
  const [job, setJob] = useState<Job | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const appendSelection = useRef(false);
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
    const urls = files.map(photo => URL.createObjectURL(photo));
    setPreviews(urls);
    return () => urls.forEach(url => URL.revokeObjectURL(url));
  }, [files]);

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

  const select = (selection: File[], append = false) => {
    if (!selection.length || busy) return;
    const next = append ? [...files, ...selection] : selection;
    if (next.length > 5) {setError('Choose up to 5 photos of the same product.'); return;}
    if (next.some(photo => !['image/jpeg', 'image/png', 'image/webp'].includes(photo.type) || photo.size > 12_000_000)) {
      setError('Choose JPG, PNG or WebP photos up to 12 MB each.'); return;
    }
    if (next.reduce((total, photo) => total + photo.size, 0) > 48_000_000) {
      setError('Maximum total upload size: 48 MB.'); return;
    }
    setFiles(next); setJob(null); setError(''); setShowHistory(false);
  };
  const pickPhotos = (append = false) => {appendSelection.current = append; input.current?.click();};
  const updatePhotos = (next: File[]) => {setFiles(next); setJob(null); setError(''); setShowHistory(false);};
  const generate = async () => {
    if (!file || busy || requestLock.current) return;
    requestLock.current = true;
    setSubmitting(true); setJob(null); setError(''); setStartedAt(Date.now()); setElapsed(0); setShowHistory(false);
    try {
      const body = new FormData(); body.append('file', file);
      files.slice(1).forEach(photo => body.append('additional_files', photo));
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
      <input ref={input} type="file" accept="image/png,image/jpeg,image/webp" multiple hidden disabled={busy} onChange={e => {select(Array.from(e.target.files || []), appendSelection.current); e.target.value = '';}}/>
      <div className="comparison">
        <section aria-label="Original photo">
          <div className="caption"><span>{files.length > 1 ? 'Main photo' : 'Original'}</span>{file && <span className="filename" title={file.name}>{file.name}</span>}</div>
          <button className={'photo upload ' + (dragging ? 'dragging' : '')} disabled={busy} aria-label={preview ? 'Change photos' : 'Upload photos'}
            onClick={() => pickPhotos()} onDragOver={e => {e.preventDefault(); if (!busy) setDragging(true);}}
            onDragLeave={() => setDragging(false)} onDrop={e => {e.preventDefault(); setDragging(false); select(Array.from(e.dataTransfer.files), files.length > 0);}}>
            {preview ? <><img src={preview} alt="Main product reference"/><span className="change">Change photos</span></> :
              <div className="empty"><Upload size={24} strokeWidth={1.5}/><strong>Upload photos</strong><span>or drop them here</span><small>Up to 5 · same product · 12 MB each</small></div>}
          </button>
          {files.length > 0 && <div className="input-views" aria-label="Product reference photos">
            <div className="view-thumbnails">{files.map((photo, index) => <div className={'view-thumbnail ' + (index === 0 ? 'selected' : '')} key={previews[index] || index}>
              <button className="view-select" disabled={busy} aria-label={index === 0 ? 'Main photo' : `Use photo ${index + 1} as main`} aria-pressed={index === 0}
                onClick={() => updatePhotos([photo, ...files.filter((_, position) => position !== index)])}>
                <img src={previews[index]} alt={photo.name}/><span>{index === 0 ? 'Main' : index + 1}</span>
              </button>
              <button className="view-remove" disabled={busy} aria-label={`Remove photo ${index + 1}`} onClick={() => updatePhotos(files.filter((_, position) => position !== index))}><X size={12}/></button>
            </div>)}
            {files.length < 5 && <button className="add-view" disabled={busy} onClick={() => pickPhotos(true)} aria-label="Add photos"><Plus size={20}/><span>Add</span></button>}</div>
            <p className="views-hint">Main photo sets the view. Add details or other angles.</p>
          </div>}
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
            {step.details.view_count && step.details.view_count > 1 && <small>{step.details.view_count} product views</small>}
            {step.details.input_image_count && step.details.input_image_count > 2 && <small>{step.details.input_image_count} reference images</small>}
            <p>{Object.entries(step.libraries).map(([name, version]) => `${name} ${version}`).join(' · ')} · Python {step.python}</p>
            {step.details.cache_hit && <small>Cached result · no model request</small>}
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
      {job?.result_url && <p className="review-note">{job.classification?.review_required
        ? 'Product identification is uncertain. Check that the result shows the intended item.'
        : job.status === 'review_required' ? 'Check colors, prints and details before publishing.' : 'Check details before publishing.'}</p>}
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

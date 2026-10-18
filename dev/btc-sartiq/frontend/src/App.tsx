import {type CSSProperties, useEffect, useRef, useState} from 'react';
import {Sparkles, Zap, Check, Camera, Upload, Download, ArrowRight, ListOrdered, Loader2, Image as ImageIcon, Plus, X, Columns2, Shirt} from 'lucide-react';
import {CameraCapture, isMobileCameraDevice} from './components/CameraCapture';
import {ComparisonView} from './components/ComparisonView';
import {EditorialReveal} from './components/EditorialReveal';
import {GenerationParticles} from './components/GenerationParticles';
import {useImageReveal} from './hooks/useImageReveal';
import {useGenerationProgress} from './hooks/useGenerationProgress';
import {startStudioJob} from './lib/studioApi';

type StepImage = {id: string; label: string; url: string};
type TraceStep = {id: string; label: string; model: string | null; libraries: Record<string, string>; python: string;
  status: string; seconds: number | null; details: {cache_hit?: boolean; resolution?: string; model_invoked?: boolean; view_count?: number; input_image_count?: number}};
type Job = {job_id: string; status: string; result_url?: string; error?: string; current_image?: StepImage; images?: StepImage[]; execution_trace?: TraceStep[];
  quality?: {single_product?: {status?: string}};
  classification?: {review_required?: boolean}; stages?: {status:string; label:string}[]};
const defaultPhases = [
  {status: 'classifying', label: 'Identifying product'},
  {status: 'locating', label: 'Locating product'},
  {status: 'preparing', label: 'Preparing'},
  {status: 'balancing_color', label: 'Balancing colors'},
  {status: 'describing', label: 'Analyzing product'},
  {status: 'generating', label: 'Creating photo'},
  {status: 'pressing', label: 'Removing wrinkles'},
  {status: 'checking_product', label: 'Checking product'},
  {status: 'finishing', label: 'Refining'},
  {status: 'checking_quality', label: 'Final check'},
];
const finished = ['complete', 'review_required', 'error'];

export default function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const file = files[0];
  const preview = previews[0];
  const [referenceSize, setReferenceSize] = useState({source: '', aspect: .75});
  const referenceAspect = referenceSize.source === preview ? referenceSize.aspect : .75;
  const [job, setJob] = useState<Job | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const reveal = useImageReveal(job?.result_url);
  const [error, setError] = useState('');
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [mobile] = useState(isMobileCameraDevice);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [mobilePanel, setMobilePanel] = useState<'input' | 'result'>('input');
  const input = useRef<HTMLInputElement>(null);
  const appendSelection = useRef(false);
  const requestLock = useRef(false);
  const generationVersion = useRef(0);
  const processing = submitting || !!(job && !finished.includes(job.status));
  const outputVisible = reveal.phase === 'completion';
  const revealing = reveal.phase === 'revealing';
  const busy = processing || !!(job?.result_url && !outputVisible && reveal.phase !== 'error');
  const currentImage = job?.result_url ? {id: 'final', label: 'Final output', url: job.result_url} : job?.current_image;
  const canViewHistory = !!job && finished.includes(job.status) && !!job.images?.length;
  const phases = job?.stages || defaultPhases;

  const phaseIndex = phases.findIndex(phase => phase.status === job?.status);
  const progressLabel = job?.result_url ? (reveal.decoded ? 'Ready' : 'Loading finished photo') : submitting ? 'Uploading' : job?.status === 'queued' ? 'Queued' : phases[phaseIndex]?.label || 'Processing';
  const progressPercent = useGenerationProgress(submitting ? 'uploading' : job?.status || 'queued', busy, !!job?.result_url && reveal.decoded, startedAt);

  useEffect(() => {
    if (!busy || startedAt === null) return;
    const tick = () => setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    tick();
    const timer = window.setInterval(tick, 1000);
    return () => window.clearInterval(timer);
  }, [busy, startedAt]);

  useEffect(() => {
    if (reveal.phase === 'error') setError('The finished photo could not be loaded. Download it or try again.');
  }, [reveal.phase]);

  useEffect(() => {
    const urls = files.map(photo => URL.createObjectURL(photo));
    setPreviews(urls);
    return () => urls.forEach(url => URL.revokeObjectURL(url));
  }, [files]);

  useEffect(() => {
    if (!job || finished.includes(job.status)) return;
    const events = new EventSource('/api/jobs/' + job.job_id + '/events');
    events.onmessage = event => {
      try {
        const next: Job = JSON.parse(event.data);
        setJob(next);
        setError(next.status === 'error' ? next.error || 'Processing failed.' : '');
        if (finished.includes(next.status)) events.close();
      } catch {setError('Unable to read progress updates.');}
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
    setFiles(next); setJob(null); setError(''); setComparing(false); setShowHistory(false); setMobilePanel('input');
  };
  const pickPhotos = (append = false) => {
    appendSelection.current = append;
    if (mobile) setCameraOpen(true);
    else input.current?.click();
  };
  const updatePhotos = (next: File[]) => {setFiles(next); setJob(null); setError(''); setComparing(false); setShowHistory(false); setMobilePanel('input');};
  const generate = async (photos = files) => {
    if (!photos.length || busy || requestLock.current) return;
    requestLock.current = true;
    const version = ++generationVersion.current;
    setSubmitting(true); setJob(null); setError(''); setStartedAt(Date.now()); setElapsed(0); setComparing(false); setShowHistory(false);
    setMobilePanel('result');
    try {
      const jobId = await startStudioJob(photos);
      if (version === generationVersion.current) setJob({job_id: jobId, status: 'queued'});
    } catch (e) {if (version === generationVersion.current) setError(e instanceof Error ? e.message : String(e));}
    finally {if (version === generationVersion.current) {setSubmitting(false); requestLock.current = false;}}
  };

  const goHome = () => {generationVersion.current++; requestLock.current = false; setSubmitting(false); setFiles([]); setJob(null); setError(''); setComparing(false); setShowHistory(false); setCameraOpen(false);};
  const download = async () => {
    if (!job?.result_url) return;
    try {
      const response = await fetch(job.result_url);
      if (!response.ok) throw new Error('Unable to download photo.');
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement('a'); link.href = url; link.download = 'sartiq-product.png'; link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch {setError('Unable to download photo. Please try again.');}
  };
  return <div className={'app ' + (busy ? 'is-generating' : job?.result_url ? 'has-result' : files.length ? 'is-preparing' : '')}>
    <header className="site-header"><a href="/" className="brand" aria-label="Sartiq home" onClick={event => {event.preventDefault(); goHome();}}>sartiq<span className="brand-dot">.</span></a></header>
    <main className="studio" onDragOver={event => {if (!mobile && !busy && event.dataTransfer.types.includes('Files')) {event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; setDragging(true);}}}
      onDragLeave={event => {if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false);}}
      onDrop={event => {if (!mobile && event.dataTransfer.files.length) {event.preventDefault(); setDragging(false); select(Array.from(event.dataTransfer.files), files.length > 0 && !job?.result_url);}}}>
      {!files.length && <section className={'editorial-showcase ' + (dragging ? 'editorial-showcase--dragging' : '')}>
        <div className="editorial-showcase__copy"><h1>Your photo.<br />A better presentation</h1></div>
        <div className="home-visual"><EditorialReveal />
          <ol className="home-pipeline" aria-label="From your photo to a finished product image">
            <li><span className="home-pipeline__icon"><Camera size={23}/><Zap className="home-pipeline__flash" size={11}/></span><span>Capture</span><ArrowRight className="home-pipeline__arrow" size={18} aria-hidden="true"/></li>
            <li><span className="home-pipeline__icon"><Sparkles size={23}/></span><span>Sartiq</span><ArrowRight className="home-pipeline__arrow" size={18} aria-hidden="true"/></li>
            <li><span className="home-pipeline__icon"><Shirt size={23}/></span><span>Product photo</span></li>
          </ol>
          <div className="home-upload"><button className="primary upload" onClick={() => pickPhotos()}>{mobile ? <Camera size={20}/> : <Upload size={20}/>}{mobile ? 'Take a product photo' : 'Upload product photos'}</button></div>
          <p className="desktop-upload-hint">Or drag your photos anywhere here.<br/><span>JPG, PNG or WebP · Up to 12 MB per photo</span></p>
        </div>
      </section>}
      <input ref={input} type="file" accept="image/png,image/jpeg,image/webp" multiple hidden disabled={busy} onChange={e => {select(Array.from(e.target.files || []), appendSelection.current); e.target.value = '';}}/>
      {files.length > 0 && <>
      <div className="desktop-workspace-heading"><h1>{busy ? 'Bringing your product into focus.' : job?.result_url ? 'Your photo, ready.' : 'Start with your product.'}</h1></div>
      <div className={'workbench ' + (dragging ? 'workbench--dragging' : '')} style={{'--reference-aspect': referenceAspect} as CSSProperties}>
      <div className="preview-tabs" aria-label="Preview selection">
        <button aria-pressed={mobilePanel === 'input'} onClick={() => setMobilePanel('input')}>Original</button>
        <button aria-pressed={mobilePanel === 'result'} disabled={!currentImage && !busy} onClick={() => setMobilePanel('result')}>{busy ? 'In progress' : 'Result'}</button>
      </div>
      <div className={"comparison " + (busy ? "generation-layout" : "")}>
        <div className="reference-workspace" data-hidden={busy || !!job?.result_url}><section className="reference-panel" data-hidden={busy || !!job?.result_url} aria-label="Original photo" data-active={mobilePanel === 'input'}>
          <div className="caption"><span>{files.length > 1 ? 'Main photo' : 'Original'}</span>{file && <span className="filename" title={file.name}>{file.name}</span>}</div>
          <button className={'photo upload ' + (dragging ? 'dragging' : '')} disabled={busy} aria-label={preview ? 'Change photos' : 'Upload photos'}
            onClick={() => pickPhotos()} onDragOver={e => {e.preventDefault(); if (!busy) setDragging(true);}}
            onDragLeave={() => setDragging(false)} onDrop={e => {e.preventDefault(); e.stopPropagation(); setDragging(false); select(Array.from(e.dataTransfer.files), files.length > 0);}}>
            {preview ? <><img src={preview} alt="Main product reference" onLoad={event => {
              const image = event.currentTarget;
              if (image.naturalWidth && image.naturalHeight) setReferenceSize({source: preview, aspect: image.naturalWidth / image.naturalHeight});
            }}/><span className="change">Change photos</span></> :
              <div className="empty"><Upload size={26} strokeWidth={1.4}/><strong>{mobile ? 'Take a photo' : 'Add product photos'}</strong>{!mobile && <span>Drop here or browse</span>}<small>JPG, PNG, WebP · 12 MB each</small></div>}
          </button>
        </section>
          {mobile && <button className="library-link" disabled={busy} onClick={() => {appendSelection.current = files.length > 0; input.current?.click();}}><ImageIcon size={16}/>Choose from photos</button>}
          <p className="desktop-reference-hint">Select a thumbnail to use it as the main reference.</p>
          {files.length > 0 && <div className="input-views" aria-label="Product reference photos">
            <div className="view-thumbnails">{files.map((photo, index) => <div className={'view-thumbnail ' + (index === 0 ? 'selected' : '')} key={previews[index] || index}>
              <button className="view-select" disabled={busy} aria-label={index === 0 ? 'Main photo' : `Use photo ${index + 1} as main`} aria-pressed={index === 0}
                onClick={() => updatePhotos([photo, ...files.filter((_, position) => position !== index)])}>
                <img src={previews[index]} alt={photo.name}/><span>{index === 0 ? 'Main' : index + 1}</span>
              </button>
              <button className="view-remove" disabled={busy} aria-label={`Remove photo ${index + 1}`} onClick={() => updatePhotos(files.filter((_, position) => position !== index))}><X size={12}/></button>
            </div>)}
            {files.length < 5 && <button className="add-view" disabled={busy} onClick={() => pickPhotos(true)} aria-label="Add photos"><Plus size={20}/><span>Add</span></button>}</div>
          </div>}
        </div>
        <div className={'pipeline-connector ' + (busy ? 'pipeline-connector--active' : '')}>
          <span className="pipeline-line" aria-hidden="true"/>
          {job?.result_url ? <span className="pipeline-complete" aria-label="Generation complete"><Check size={20}/></span> :
            <button className="primary pipeline-generate" disabled={busy} onClick={() => void generate()}>
              {busy ? <><Loader2 className="spinner" size={16}/>Generating</> : <>Generate<ArrowRight size={16}/></>}
            </button>}
        </div>
        <div className="output-workspace"><section className="result-panel" data-empty={!busy && !job?.result_url} aria-label="Current image" aria-busy={busy} data-active={mobilePanel === 'result'}>
          <div className="caption"><span>{busy ? 'Live preview' : 'Result'}</span>{outputVisible && <span className="ready">{job?.quality?.single_product?.status === 'unverified' ? 'Review' : 'Ready'}</span>}</div>
          <div className={'photo result ' + (currentImage?.id === 'cutout' ? 'transparent ' : '') + (busy ? 'processing ' : '') + (revealing ? 'is-revealing ' : '') + (outputVisible ? 'finished' : '')}>
            {comparing && preview && job?.result_url ? <ComparisonView original={preview} result={job.result_url}/> : <>
              <img ref={reveal.image} className={'result-final ' + (reveal.decoded ? 'result-final--visible' : '')} src={job?.result_url} alt="Final output" aria-hidden={!reveal.decoded}/>
              <div ref={reveal.overlay} className="generation-overlay" data-phase={reveal.phase} aria-hidden={outputVisible} hidden={!busy && !job?.result_url && !currentImage}>
                <img className="generation-source" src={preview} alt="Original product being transformed"/>
                {busy && <><GenerationParticles source={preview}/><span className="generation-grain" aria-hidden="true"/><span className="live-caption" role="progressbar" aria-label="Image creation progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progressPercent} aria-valuetext={`${progressPercent}% estimated progress, ${elapsed} seconds elapsed`}>
                  <span className="generation-status-dot"/><span className="generation-stage">{progressPercent}% · {progressLabel}</span><span className="generation-elapsed" aria-label="Elapsed time">{Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, '0')}</span>
                </span></>}
              </div>
              {!busy && !job?.result_url && !currentImage && <div className="empty"><Shirt size={40} strokeWidth={1.1}/><strong>A new perspective on your product.</strong><span>Your finished photo will appear here.</span><ol className="desktop-preparation"><li><b>01</b>Choose your references</li><li><b>02</b>Generate your photo</li><li><b>03</b>Compare and download</li></ol></div>}
            </>}

          </div>
        </section>
      <div className="actions" hidden={busy}>
        {!job?.result_url && <button className="primary mobile-generate" onClick={() => void generate()} disabled={!file || busy}>{busy ? <><Loader2 className="spinner" size={16}/>Processing</> : <>Generate photo<ArrowRight size={17}/></>}</button>}
        {job?.result_url && <>
          <button className="primary" onClick={() => void download()}><Download size={18}/>Download</button>
          <button className="secondary" aria-pressed={comparing} onClick={() => setComparing(value => !value)}><Columns2 size={18}/>Compare</button>
        </>}
        {canViewHistory && <button className="history-toggle" aria-expanded={showHistory} aria-controls="image-history" onClick={() => setShowHistory(value => !value)}><ListOrdered size={18}/>Steps</button>}
      </div>
      </div></div>
      </div>
      {canViewHistory && showHistory && <section id="image-history" className="image-history" aria-label="Intermediate images">
        {job?.images?.map((image, index) => <figure key={image.id}>
          <a className={'history-image ' + (image.id === 'cutout' ? 'transparent' : '')} href={image.url} target="_blank" rel="noreferrer" aria-label={'Open ' + image.label}>
            <img src={image.url} alt={image.label} loading="lazy"/>
          </a>
          <figcaption><span>{String(index + 1).padStart(2, '0')}</span>{image.label}</figcaption><p>{stepDescription(image.id)}</p>
        </figure>)}
      </section>}
      </>}
      {error && <p className="error" role="alert">{error}</p>}
    </main>
    {cameraOpen && mobile && <CameraCapture onClose={() => setCameraOpen(false)}
      onCapture={photos => {select(photos); setCameraOpen(false); void generate(photos);}}
      onPhotos={photos => {select(photos); setCameraOpen(false);}}/>}
  </div>;
}

function stepDescription(id: string) {
  const descriptions: Record<string, string> = {
    original: 'The original photograph provides the product shape, colors and details.',
    cutout: 'The garment is isolated from the surrounding background.',
    target: 'The product is located in the source photograph; the outline shows the selected area.',
    mask: 'A precise silhouette separates the selected product from its surroundings.',
    segmented: 'The selected product is placed against white to remove background distractions.',
    color: 'Uneven illumination is corrected using the selected product area.',
    reference: 'The corrected product and clean background form the reference for generation.',
    description: 'A close crop is used to analyze fabric, shape, color and design details.',
    generated: 'A studio composition is created using the product references.',
    pressed: 'Wrinkles are softened while retaining the garment’s structure.',
    composed: 'The isolated product is centered on a studio background with consistent framing.',
    final: 'The finished product image brings the processing stages together.',
  };
  return descriptions[id] || (id.startsWith('original_') ? 'An additional view supplies product details alongside the main photograph.' : '') || 'This intermediate image shows the product at this stage of preparation.';
}

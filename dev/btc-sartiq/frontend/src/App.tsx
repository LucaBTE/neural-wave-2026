import {useCallback, useEffect, useRef, useState} from 'react';
import {AlertCircle, ArrowLeft, Camera, Check, Download, Image as ImageIcon, RefreshCw, ScanLine, Share2, Sparkles, Upload} from 'lucide-react';
import {ComparisonView} from './components/ComparisonView';
import {GlassButton} from './components/GlassButton';
import {TransformReveal} from './components/TransformReveal';
import {useStudioJob} from './hooks/useStudioJob';

type View = 'landing' | 'camera' | 'upload' | 'processing' | 'reveal' | 'result' | 'compare';

function CameraStage({onCapture, onBack}: {onCapture: (file: File) => void; onBack: () => void}) {
  const video = useRef<HTMLVideoElement>(null);
  const picker = useRef<HTMLInputElement>(null);
  const [message, setMessage] = useState('Opening camera…');
  const [ready, setReady] = useState(false);
  const [capturing, setCapturing] = useState(false);

  useEffect(() => {
    let stream: MediaStream | undefined;
    let disposed = false;
    let cameraReady = false;
    const fallbackTimer = window.setTimeout(() => {
      if (!cameraReady) setMessage('Camera is taking too long. You can choose a photo instead.');
    }, 4000);
    const openCamera = async () => {
      if (!navigator.mediaDevices?.getUserMedia) { setMessage('Camera is not available on this device.'); return; }
      try {
        stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: {ideal: 'environment'}}, audio: false});
        if (disposed) { stream.getTracks().forEach(track => track.stop()); return; }
        const source = video.current;
        if (!source) return;
        const markReady = () => {
          if (!disposed && source.videoWidth > 0 && source.videoHeight > 0) {
            cameraReady = true;
            window.clearTimeout(fallbackTimer);
            setReady(true);
            setMessage('Keep the garment fully in frame');
          }
        };
        source.srcObject = stream;
        source.addEventListener('loadedmetadata', markReady, {once: true});
        source.addEventListener('loadeddata', markReady, {once: true});
        await source.play();
        markReady();
      } catch { setMessage('Camera access was not granted. You can still choose a photo.'); }
    };
    void openCamera();
    return () => { disposed = true; window.clearTimeout(fallbackTimer); stream?.getTracks().forEach(track => track.stop()); };
  }, []);

  const capture = () => {
    const source = video.current;
    if (!source || !source.videoWidth || !source.videoHeight) {
      setMessage('Camera is still getting ready. Please try again in a moment.');
      return;
    }
    setCapturing(true);
    setMessage('Capturing your garment…');
    const canvas = document.createElement('canvas');
    canvas.width = source.videoWidth; canvas.height = source.videoHeight;
    canvas.getContext('2d')?.drawImage(source, 0, 0);
    canvas.toBlob(blob => {
      if (blob) onCapture(new File([blob], `neural-wave-${Date.now()}.jpg`, {type: 'image/jpeg'}));
      else { setCapturing(false); setMessage('We could not capture that frame. Please try again.'); }
    }, 'image/jpeg', 0.94);
  };

  return <section className="camera-stage">
    <video ref={video} muted playsInline aria-label="Live camera preview" />
    <div className="camera-stage__veil" aria-hidden="true" />
    <div className="camera-stage__top"><GlassButton tone="dark" onClick={onBack} aria-label="Go back"><ArrowLeft size={20}/></GlassButton><span className="camera-stage__title">Frame your garment</span><span className="camera-stage__top-spacer" /></div>
    <div className="lens-guide" aria-hidden="true"><span /><span /><span /><span /><i><ScanLine size={20}/></i></div>
    <div className="camera-stage__bottom"><p>{ready ? 'Keep the garment fully in frame' : message}</p><div className="camera-stage__controls">
      <GlassButton tone="dark" onClick={() => picker.current?.click()} aria-label="Choose a photo"><ImageIcon size={20}/></GlassButton>
      <button className="shutter" disabled={!ready || capturing} onClick={capture} aria-label="Take photo"><span /></button><span className="camera-stage__control-spacer" />
    </div></div>
    <input ref={picker} hidden type="file" accept="image/jpeg,image/png,image/webp" onChange={event => { const picked = event.target.files?.[0]; if (picked) onCapture(picked); event.target.value = ''; }}/>
  </section>;
}

export default function App() {
  const [compact, setCompact] = useState(() => window.matchMedia('(max-width: 720px)').matches);
  const [view, setView] = useState<View>(() => window.matchMedia('(max-width: 720px)').matches ? 'landing' : 'upload');
  const [loadedResult, setLoadedResult] = useState('');
  const [shareMessage, setShareMessage] = useState('');
  const upload = useRef<HTMLInputElement>(null);
  const studio = useStudioJob();
  const result = studio.job?.result_url || '';

  useEffect(() => {
    const media = window.matchMedia('(max-width: 720px)'); const change = () => setCompact(media.matches);
    media.addEventListener('change', change); return () => media.removeEventListener('change', change);
  }, []);
  useEffect(() => {
    if (!result) { setLoadedResult(''); return; }
    let active = true; const image = new Image();
    image.onload = () => { if (active) setLoadedResult(result); }; image.onerror = () => { if (active) setLoadedResult(result); }; image.src = result;
    return () => { active = false; };
  }, [result]);
  useEffect(() => { if (loadedResult && result === loadedResult && view === 'processing') setView('reveal'); }, [loadedResult, result, view]);

  const begin = async (candidate?: File) => {
    if (candidate && !studio.selectFile(candidate)) return;
    if (candidate || studio.file) setView('processing');
    const started = await studio.generate(candidate);
    if (started) setView('processing');
  };
  const chooseDesktop = (candidate?: File) => { if (candidate) studio.selectFile(candidate); };
  const closeToStart = () => setView(compact ? 'landing' : 'upload');
  const revealComplete = useCallback(() => setView('result'), []);
  const download = () => {
    if (!result) return; const link = document.createElement('a');
    link.href = result; link.download = 'neural-wave-product-image.png'; link.target = '_blank'; link.rel = 'noreferrer'; link.click();
  };
  const share = async () => {
    if (!result) return;
    try {
      const response = await fetch(result); const blob = await response.blob();
      const file = new File([blob], 'neural-wave-product-image.png', {type: blob.type || 'image/png'});
      if (navigator.share && (!navigator.canShare || navigator.canShare({files: [file]}))) await navigator.share({title: 'Neural Wave product image', files: [file]});
      else if (navigator.clipboard) { await navigator.clipboard.writeText(result); setShareMessage('Link copied'); }
      else window.open(result, '_blank', 'noopener,noreferrer');
    } catch { setShareMessage('Share was cancelled'); }
    window.setTimeout(() => setShareMessage(''), 1800);
  };

  if (view === 'compare' && studio.preview && result) return <ComparisonView original={studio.preview} result={result} onClose={() => setView('result')} />;
  if (view === 'camera') return <CameraStage onCapture={file => void begin(file)} onBack={() => setView('landing')} />;

  if (view === 'landing') return <main className="mobile-landing">
    <div className="mobile-landing__orb" aria-hidden="true" />
    <header className="brand-bar"><span>neural<span>.</span>wave</span><small>Studio</small></header>
    <section className="mobile-landing__hero"><p className="eyebrow">Product photography, reimagined</p><h1>Your garment.<br />Ready for retail.</h1><p>Turn one photo into a polished product image in moments.</p></section>
    <div className="mobile-landing__visual" aria-hidden="true"><span className="mobile-landing__ring" /><Sparkles size={25}/></div>
    <GlassButton className="capture-cta" onClick={() => setView('camera')}><Camera size={20}/><span>Photograph garment</span><i /></GlassButton>
    <button className="text-action" onClick={() => upload.current?.click()}><Upload size={15}/> Choose from library</button>
    <input ref={upload} hidden type="file" accept="image/jpeg,image/png,image/webp" onChange={event => { const picked = event.target.files?.[0]; if (picked) void begin(picked); event.target.value = ''; }}/>
  </main>;

  if (view === 'processing' || view === 'reveal') return <main className="process-screen">
    <header className="process-screen__header"><button onClick={closeToStart} aria-label="Back to image selection"><ArrowLeft size={21}/></button><span>Neural Wave</span><span /></header>
    <section className="process-card">
      {view === 'reveal' && studio.preview && result ? <TransformReveal original={studio.preview} result={result} onComplete={revealComplete} /> : <>
        <div className="process-image"><img src={studio.preview} alt="Selected garment" /><div className="process-image__scan" /></div>
        <div className="process-copy"><span className="processing-mark"><Sparkles size={16}/></span><h1>{studio.error ? 'Something interrupted the process' : 'Creating your product image'}</h1><p>{studio.error || 'We’re refining the photo while preserving the character of your garment.'}</p></div>
        {studio.error ? <GlassButton tone="light" onClick={() => void begin()}><RefreshCw size={17}/> Try again</GlassButton> : <div className="process-progress"><i /><i /><i /></div>}
      </>}
    </section>
  </main>;

  if (view === 'result' && result) return <main className="result-screen">
    <header className="result-screen__header"><GlassButton tone="dark" onClick={closeToStart} aria-label="Back to image selection"><ArrowLeft size={20}/></GlassButton><span>Finished image</span><span /></header>
    <section className="result-screen__image"><img src={result} alt="Finished professional product photograph" /></section>
    <div className="result-screen__copy"><span><Check size={14}/> Ready for retail</span><h1>A polished product image.</h1></div>
    <nav className="result-actions" aria-label="Image actions"><button onClick={download} aria-label="Download image"><Download size={19}/><span>Download</span></button><button onClick={() => void share()} aria-label="Share image"><Share2 size={19}/><span>Share</span></button><button onClick={() => setView('compare')} aria-label="Compare images"><ScanLine size={19}/><span>Compare</span></button></nav>
    {shareMessage && <span className="toast" role="status">{shareMessage}</span>}
  </main>;

  return <main className="desktop-studio">
    <header className="desktop-studio__header"><span className="desktop-brand">neural<span>.</span>wave</span><span className="desktop-studio__label">Product image studio</span></header>
    <section className="desktop-intro"><p className="eyebrow">Neural Wave Studio</p><h1>Make every garment<br />look ready for retail.</h1><p>Upload a photo and let Neural Wave create a clean, considered product image.</p></section>
    <section className="desktop-workspace">
      <div className="image-panel image-panel--input"><div className="panel-caption"><span>Original</span>{studio.file && <button onClick={studio.reset}>Remove</button>}</div><button className="upload-area" onClick={() => upload.current?.click()} onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); chooseDesktop(event.dataTransfer.files[0]); }}>
        {studio.preview ? <><img src={studio.preview} alt="Selected garment"/><span className="replace-image">Replace photo</span></> : <><Upload size={25}/><strong>Upload a garment photo</strong><small>Drop it here or browse your files<br />JPG, PNG, WebP · max 12 MB</small></>}
      </button></div>
      <div className="image-panel image-panel--output"><div className="panel-caption"><span>Result</span><span className="panel-caption__status">Ready when you are</span></div><div className="output-placeholder"><Sparkles size={28}/><strong>Your finished image will appear here.</strong><span>Clean, focused and ready for your catalogue.</span></div></div>
    </section>
    {studio.error && <p className="desktop-error"><AlertCircle size={17}/>{studio.error}</p>}
    <div className="desktop-studio__footer"><GlassButton tone="light" disabled={!studio.file || studio.busy} onClick={() => void begin()}><Sparkles size={17}/>{studio.busy ? 'Working…' : 'Create product image'}</GlassButton></div>
    <input ref={upload} hidden type="file" accept="image/jpeg,image/png,image/webp" onChange={event => { chooseDesktop(event.target.files?.[0]); event.target.value = ''; }}/>
  </main>;
}

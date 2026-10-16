import {useEffect, useRef, useState} from 'react';
import {Camera, UploadCloud, Sparkles, Download, ArrowRight} from 'lucide-react';

type Job = {job_id: string; status: string; result_url?: string; error?: string};
const finished = ['complete', 'review_required', 'error'];

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const input = useRef<HTMLInputElement>(null);
  const requestLock = useRef(false);
  const busy = submitting || !!(job && !finished.includes(job.status));

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!job || finished.includes(job.status)) return;
    const controller = new AbortController();
    let stopped = false;
    let timer: number;
    const poll = async () => {
      try {
        const response = await fetch('/api/jobs/' + job.job_id, {signal: controller.signal});
        const data = await response.json();
        if (!response.ok) throw Error(data.detail || 'Impossibile leggere il risultato');
        if (stopped) return;
        setJob(data);
        setError(data.status === 'error' ? data.error : '');
        if (finished.includes(data.status)) return;
      } catch (e) {
        if (stopped) return;
        setError('Connessione interrotta. Riprovo a recuperare il risultato…');
      }
      timer = window.setTimeout(poll, 1800);
    };
    void poll();
    return () => {stopped = true; controller.abort(); window.clearTimeout(timer);};
  }, [job?.job_id]);

  const select = (next?: File) => {
    if (!next || busy) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(next.type) || next.size > 12_000_000) {
      setError('Scegli una foto JPG, PNG o WebP fino a 12 MB.'); return;
    }
    setFile(next); setJob(null); setError('');
  };
  const generate = async () => {
    if (!file || busy || requestLock.current) return;
    requestLock.current = true;
    setSubmitting(true); setJob(null); setError('');
    try {
      const body = new FormData(); body.append('file', file);
      const response = await fetch('/api/studio', {method: 'POST', body});
      const data = await response.json();
      if (!response.ok) throw Error(data.detail || 'Impossibile avviare la generazione');
      setJob({job_id: data.job_id, status: 'queued'});
    } catch (e) {setError(e instanceof Error ? e.message : String(e));}
    finally {setSubmitting(false); requestLock.current = false;}
  };

  return <div className="app">
    <header><a className="brand" href="/">SARTIQ <span className="brand-slash">/</span> <span className="brand-small">CATALOGUE STUDIO</span></a></header>
    <main>
      <div className="intro"><div><p className="eyebrow">IL TUO STUDIO FOTOGRAFICO</p><h1>Da una foto qualsiasi<br/>al tuo <em>catalogo.</em></h1><p className="lede">Sfondo bianco, luce da studio e un capo disposto con cura. Carica la foto: al resto pensiamo noi.</p></div></div>
      <div className="workspace">
        <section className="controls"><div className="panel">
          <h2>La tua foto</h2>
          <input ref={input} type="file" accept="image/png,image/jpeg,image/webp" hidden disabled={busy} onChange={e => select(e.target.files?.[0])}/>
          <button className={'drop ' + (preview ? 'has-file' : '')} disabled={busy} onClick={() => input.current?.click()} onDragOver={e => e.preventDefault()} onDrop={e => {e.preventDefault(); select(e.dataTransfer.files[0]);}}>
            {preview ? <><img src={preview} alt="Foto caricata"/><div className="image-cover">Cambia foto</div></> : <><UploadCloud size={33}/><strong>Trascina qui la foto del capo</strong><span>oppure clicca · JPG, PNG, WebP · max 12 MB</span></>}
          </button>
          <p className="hint">Un solo capo ben visibile, con stampe e dettagli a fuoco.</p>
          <button className="primary generate" onClick={generate} disabled={!file || busy}><Sparkles size={17}/>{busy ? 'Creo la tua foto…' : 'Crea foto da catalogo'}<ArrowRight size={16}/></button>
          <p className="hint">L’elaborazione può richiedere alcuni minuti.</p>
        </div></section>
        <section className="results" aria-live="polite">
          <div className="results-heading"><h2>Prima e dopo</h2><span className="status">{busy ? 'IN ELABORAZIONE' : job?.result_url ? 'PRONTA' : 'CARICA UNA FOTO'}</span></div>
          <div className="image-grid">
            <div className="image-card"><div className="image-area">{preview ? <img src={preview} alt="Prima: foto originale"/> : <div className="placeholder"><Camera size={30}/><span>La tua foto originale</span></div>}</div><div className="image-caption">PRIMA</div></div>
            <div className="image-card"><div className="image-area">{job?.result_url ? <img src={job.result_url} alt="Dopo: capo su sfondo bianco da studio"/> : <div className="placeholder"><Sparkles size={30}/><span>{busy ? 'Prepariamo il tuo scatto da studio…' : 'La foto da catalogo apparirà qui'}</span></div>}</div><div className="image-caption">DOPO</div></div>
          </div>
          {error && <div className="error" role="alert">{error}</div>}
          {job?.result_url && <><a className="download" href={job.result_url} target="_blank" rel="noreferrer"><Download size={16}/>Apri la foto finale</a><p className="hint">{job.status === 'review_required' ? 'Controlla la resa dei colori rispetto al capo originale. ' : ''}Confronta sempre scritte, stampe e dettagli prima di pubblicare.</p></>}
        </section>
      </div>
    </main><footer><span>SARTIQ — CATALOGUE STUDIO</span><span>DAL TUO CAPO ALLO SCATTO DA STUDIO</span></footer>
  </div>;
}

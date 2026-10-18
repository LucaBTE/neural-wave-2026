import {type CSSProperties, useEffect, useRef, useState} from 'react';
import {Camera, Image as ImageIcon, X, ArrowRight} from 'lucide-react';

export function isMobileCameraDevice() {
  // Width alone would open a webcam on a resized desktop browser.
  return /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ||
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
}

type ZoomRange = {min: number; max: number; step: number};
type ZoomCapabilities = MediaTrackCapabilities & {zoom?: ZoomRange};
type ZoomSettings = MediaTrackSettings & {zoom?: number};
type ZoomConstraint = MediaTrackConstraintSet & {zoom: number};

type Props = {onCapture: (files: File[]) => void; onPhotos: (files: File[]) => void;
  onClose: () => void};

export function CameraCapture({onCapture, onPhotos, onClose}: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const readyRef = useRef(false);
  const [ready, setReady] = useState(false);
  const [unavailable, setUnavailable] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [multiple, setMultiple] = useState(false);
  const [photos, setPhotos] = useState<File[]>([]);
  const [urls, setUrls] = useState<string[]>([]);
  const [review, setReview] = useState(false);
  const [error, setError] = useState('');
  const track = useRef<MediaStreamTrack | null>(null);
  const zoomLock = useRef(false);
  const [zoomRange, setZoomRange] = useState<ZoomRange | null>(null);
  const [zoom, setZoom] = useState(1);
  const [adjustingZoom, setAdjustingZoom] = useState(false);
  const [controlError, setControlError] = useState('');
  const libraryInput = useRef<HTMLInputElement>(null);
  const nativeInput = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const next = photos.map(photo => URL.createObjectURL(photo)); setUrls(next);
    return () => next.forEach(url => URL.revokeObjectURL(url));
  }, [photos]);
  const addPhoto = (photo: File) => {
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(photo.type) || photo.size > 12_000_000) {
      setError('Choose JPG, PNG or WebP up to 12 MB.'); return;
    }
    if (photos.reduce((sum, item) => sum + item.size, photo.size) > 48_000_000) {
      setError('Maximum total upload size: 48 MB.'); return;
    }
    setError('');
    if (multiple) setPhotos(previous => [...previous, photo].slice(0, 5));
    else onCapture([photo]);
  };


  useEffect(() => {
    dialog.current?.showModal();
    dialog.current?.focus({preventScroll: true});
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {document.body.style.overflow = previousOverflow;};
  }, []);

  useEffect(() => {
    let stream: MediaStream | undefined;
    let disposed = false;
    readyRef.current = false; setReady(false); setUnavailable(false);
    setZoomRange(null); setAdjustingZoom(false); zoomLock.current = false;
    const timer = window.setTimeout(() => {if (!disposed && !readyRef.current) setUnavailable(true);}, 8000);
    const start = async () => {
      if (!isMobileCameraDevice() || !navigator.mediaDevices?.getUserMedia) {
        setUnavailable(true);
        window.clearTimeout(timer);
        return;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {facingMode: {ideal: 'environment'}, width: {ideal: 1920}, height: {ideal: 2560}}, audio: false,
        });
        if (disposed) {stream.getTracks().forEach(track => track.stop()); return;}
        const element = video.current;
        if (!element) return;
        element.srcObject = stream;
        const cameraTrack = stream.getVideoTracks()[0];
        track.current = cameraTrack;
        await element.play();
        if (disposed) return;
        // Zoom values are device-reported; do not invent a 0.5× preset.
        try {
          const range = (cameraTrack.getCapabilities?.() as ZoomCapabilities | undefined)?.zoom;
          if (range && Number.isFinite(range.min) && Number.isFinite(range.max) && range.min > 0 && range.max > range.min) {
            setZoomRange({...range, step: range.step > 0 ? range.step : .1});
            setZoom((cameraTrack.getSettings() as ZoomSettings).zoom ?? range.min);
          }
        } catch { /* Optional zoom capabilities must not interrupt the live preview. */ }
      } catch {
        if (!disposed) {
          setUnavailable(true);
        }
        stream?.getTracks().forEach(track => track.stop());
        window.clearTimeout(timer);
      }
    };
    void start();
    return () => {
      disposed = true;
      window.clearTimeout(timer);
      stream?.getTracks().forEach(track => track.stop());
      if (track.current === stream?.getVideoTracks()[0]) track.current = null;
    };
  }, []);

  const changeZoom = async (value: number) => {
    const cameraTrack = track.current;
    if (!cameraTrack || !zoomRange || zoomLock.current || capturing) return;
    const clamped = Math.min(zoomRange.max, Math.max(zoomRange.min, value));
    const next = Math.min(zoomRange.max, zoomRange.min + Math.round((clamped - zoomRange.min) / zoomRange.step) * zoomRange.step);
    zoomLock.current = true; setAdjustingZoom(true); setControlError('');
    try {
      const constraints = cameraTrack.getConstraints();
      await cameraTrack.applyConstraints({...constraints, advanced: [...(constraints.advanced || []).filter(item => !('zoom' in item)), {zoom: next} as ZoomConstraint]});
      if (track.current === cameraTrack) {
        const actual = (cameraTrack.getSettings() as ZoomSettings).zoom;
        if (actual === undefined || Math.abs(actual - next) > zoomRange.step / 2 + .001) {
          setControlError('This zoom level is not available. Try another zoom level.');
        }
        if (actual !== undefined) setZoom(actual);
      }
    } catch {
      if (track.current === cameraTrack) setControlError('Unable to adjust zoom on this camera. Try another zoom level.');
    } finally {
      if (track.current === cameraTrack) {zoomLock.current = false; setAdjustingZoom(false);}
    }
  };

  const capture = () => {
    const source = video.current;
    if (!ready || !source || capturing || adjustingZoom) return;
    setCapturing(true);
    const canvas = document.createElement('canvas');
    // Match the centered object-fit: cover crop shown in the camera frame.
    const frame = source.getBoundingClientRect();
    if (!frame.width || !frame.height) {setCapturing(false); return;}
    const scale = Math.max(frame.width / source.videoWidth, frame.height / source.videoHeight);
    const cropWidth = Math.min(source.videoWidth, frame.width / scale);
    const cropHeight = Math.min(source.videoHeight, frame.height / scale);
    canvas.width = Math.round(cropWidth);
    canvas.height = Math.round(cropHeight);
    const context = canvas.getContext('2d');
    if (!context) {setCapturing(false); setUnavailable(true); return;}
    context.drawImage(source, (source.videoWidth - cropWidth) / 2, (source.videoHeight - cropHeight) / 2,
      cropWidth, cropHeight, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
      setCapturing(false);
      if (blob) addPhoto(new File([blob], `photo-${Date.now()}.jpg`, {type: 'image/jpeg'}));
      else setUnavailable(true);
    }, 'image/jpeg', .94);
  };

  const zoomSteps = zoomRange ? Array.from(new Set([zoomRange.min, .5, 1, 2]))
    .filter(value => value !== 3 && value >= zoomRange.min && value <= zoomRange.max &&
      Math.abs((value - zoomRange.min) / zoomRange.step - Math.round((value - zoomRange.min) / zoomRange.step)) < .001)
    .sort((a, b) => a - b) : [];
  const selectedZoom = zoomSteps.findIndex(value => zoomRange && Math.abs(zoom - value) < zoomRange.step / 2 + .001);

  return <dialog ref={dialog} className="camera-screen" tabIndex={-1} aria-label="Take a product photo"
    onCancel={event => {event.preventDefault(); onClose();}}>
    <input ref={nativeInput} type="file" accept="image/jpeg,image/png,image/webp" capture="environment" hidden onChange={event => {const photo = event.target.files?.[0]; if (photo) addPhoto(photo); event.target.value = '';}}/>
    <input ref={libraryInput} type="file" accept="image/jpeg,image/png,image/webp" multiple hidden
      onChange={event => {const selected = Array.from(event.target.files || []); if (selected.length) onPhotos(selected); event.target.value = '';}}/>
    <div className="camera-header"><span className="brand">sartiq<span className="brand-dot">.</span></span>
      <div className="camera-controls"><div className="camera-mode" style={{'--selection-index': multiple ? 1 : 0} as CSSProperties} role="group" aria-label="Capture mode">
        <button aria-pressed={!multiple} disabled={photos.length > 0} onClick={() => setMultiple(false)}>Single photo</button>
        <button aria-pressed={multiple} onClick={() => setMultiple(true)}>Multi-photos</button>
      </div><button className="icon-button" onClick={onClose} aria-label="Close camera"><X size={21}/></button></div></div>
    <div className="camera-viewport">
      <video ref={video} autoPlay playsInline muted onLoadedData={() => {
        if (video.current?.videoWidth) {readyRef.current = true; setReady(true); setUnavailable(false);}
      }}/>
      {ready ? <div className="camera-guide" aria-hidden="true"/> : <div className="camera-placeholder">
        <Camera size={32} strokeWidth={1.3}/><p>{unavailable ? 'Use your camera or choose a photo.' : 'Opening camera…'}</p>
        {unavailable && <button className="secondary" disabled={photos.length >= 5} onClick={() => nativeInput.current?.click()}>Take photo</button>}
      </div>}
      {ready && zoomRange && <div className="camera-zoom" style={{'--selection-index': Math.max(0, selectedZoom), '--selection-opacity': selectedZoom < 0 ? 0 : 1} as CSSProperties} role="group" aria-label="Camera zoom">
        {zoomSteps.map(value => <button key={value} type="button"
            aria-label={`Zoom ${Number(value.toFixed(2))}×`} aria-pressed={Math.abs(zoom - value) < zoomRange.step / 2 + .001}
            disabled={capturing || adjustingZoom} onClick={() => void changeZoom(value)}>{Number(value.toFixed(2))}×</button>)}
      </div>}
      {controlError && <p className="camera-control-error" role="status">{controlError}</p>}
    </div>
    {error && <p className="error" role="alert">{error}</p>}
    {multiple && review && <div className="camera-review">{urls.map((url, index) => <div key={url}><img src={url} alt={`Captured photo ${index + 1}`}/><button aria-label={`Remove captured photo ${index + 1}`} onClick={() => setPhotos(previous => previous.filter((_, i) => i !== index))}><X size={14}/></button></div>)}</div>}
    {multiple && photos.length > 0 && <button className="primary camera-confirm" onClick={() => onCapture(photos)}>Generate from {photos.length} {photos.length === 1 ? 'photo' : 'photos'}<ArrowRight size={18}/></button>}
    <div className="camera-footer">
      <button className="camera-library" onClick={() => libraryInput.current?.click()}><ImageIcon size={22}/><span>Photos</span></button>
      <button className="shutter" onClick={capture} disabled={!ready || capturing || adjustingZoom || photos.length >= 5} aria-label="Take photo"/>
      {multiple ? <button className="camera-deck" disabled={!photos.length} aria-label={`Review ${photos.length} photos`} aria-expanded={review} onClick={() => setReview(value => !value)}>
        {urls.length ? <img src={urls[urls.length - 1]} alt="Latest captured photo"/> : <ImageIcon size={24}/>}
        <span className="camera-badge">{photos.length}</span>
      </button> : <span className="camera-spacer" aria-hidden="true"/>}
    </div>
  </dialog>;
}

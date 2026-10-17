import {useEffect, useRef, useState} from 'react';
import {Camera, Image as ImageIcon, X} from 'lucide-react';

export function isMobileCameraDevice() {
  // Width alone would open a webcam on a resized desktop browser.
  return /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ||
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
}

type Props = {onCapture: (file: File) => void; onPhotos: () => void;
  onNativeCamera: () => void; onClose: () => void};

export function CameraCapture({onCapture, onPhotos, onNativeCamera, onClose}: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const readyRef = useRef(false);
  const [ready, setReady] = useState(false);
  const [unavailable, setUnavailable] = useState(false);
  const [capturing, setCapturing] = useState(false);

  useEffect(() => {
    dialog.current?.showModal();
    let stream: MediaStream | undefined;
    let disposed = false;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
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
        await element.play();
      } catch {
        if (!disposed) setUnavailable(true);
        stream?.getTracks().forEach(track => track.stop());
        window.clearTimeout(timer);
      }
    };
    void start();
    return () => {
      disposed = true;
      window.clearTimeout(timer);
      stream?.getTracks().forEach(track => track.stop());
      document.body.style.overflow = previousOverflow;
    };
  }, []);

  const capture = () => {
    const source = video.current;
    if (!ready || !source || capturing) return;
    setCapturing(true);
    const canvas = document.createElement('canvas');
    canvas.width = source.videoWidth;
    canvas.height = source.videoHeight;
    const context = canvas.getContext('2d');
    if (!context) {setCapturing(false); setUnavailable(true); return;}
    context.drawImage(source, 0, 0);
    canvas.toBlob(blob => {
      setCapturing(false);
      if (blob) onCapture(new File([blob], `photo-${Date.now()}.jpg`, {type: 'image/jpeg'}));
      else setUnavailable(true);
    }, 'image/jpeg', .94);
  };

  return <dialog ref={dialog} className="camera-screen" aria-label="Take a product photo"
    onCancel={event => {event.preventDefault(); onClose();}}>
    <div className="camera-header"><span className="brand">sartiq<span className="brand-dot">.</span></span>
      <button className="icon-button" onClick={onClose} aria-label="Close camera"><X size={21}/></button></div>
    <div className="camera-viewport">
      <video ref={video} autoPlay playsInline muted onLoadedData={() => {
        if (video.current?.videoWidth) {readyRef.current = true; setReady(true); setUnavailable(false);}
      }}/>
      {ready ? <div className="camera-guide" aria-hidden="true"/> : <div className="camera-placeholder">
        <Camera size={32} strokeWidth={1.3}/><p>{unavailable ? 'Use your camera or choose a photo.' : 'Opening camera…'}</p>
        {unavailable && <button className="secondary" onClick={onNativeCamera}>Take photo</button>}
      </div>}
    </div>
    <div className="camera-footer">
      <button className="camera-library" onClick={onPhotos}><ImageIcon size={22}/><span>Photos</span></button>
      <button className="shutter" onClick={capture} disabled={!ready || capturing} aria-label="Take photo"/>
      <span className="camera-spacer" aria-hidden="true"/>
    </div>
  </dialog>;
}

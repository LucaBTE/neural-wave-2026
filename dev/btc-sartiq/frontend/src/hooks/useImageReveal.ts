import {useEffect, useRef, useState} from 'react';

type Phase = 'generation' | 'preloading' | 'decoding' | 'revealing' | 'completion' | 'error';

/** Decode the displayed element itself; completion belongs to the overlay animation. */
export function useImageReveal(url: string | undefined) {
  const image = useRef<HTMLImageElement>(null);
  const overlay = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<{url?: string; phase: Phase}>({phase: 'generation'});
  const phase = state.url === url ? state.phase : url ? 'preloading' : 'generation';

  useEffect(() => {
    if (!url) {setState({phase: 'generation'}); return;}
    const element = image.current;
    if (!element) return;
    let canceled = false;
    let decoding = false;
    const update = (phase: Phase) => {if (!canceled) setState({url, phase});};
    const fail = () => update('error');
    const ready = async () => {
      if (decoding) return;
      decoding = true;
      update('decoding');
      try {
        if (typeof element.decode === 'function') await element.decode();
        if (!element.naturalWidth || !element.naturalHeight) throw new Error('Invalid image');
        update('revealing');
      } catch {fail();}
    };
    update('preloading');
    element.addEventListener('load', ready);
    element.addEventListener('error', fail);
    if (element.complete) {
      if (element.naturalWidth) void ready(); else fail();
    }
    return () => {
      canceled = true;
      element.removeEventListener('load', ready);
      element.removeEventListener('error', fail);
    };
  }, [url]);

  useEffect(() => {
    if (phase !== 'revealing') return;
    const layer = overlay.current;
    if (!layer) return;
    let canceled = false;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    const animation = layer.animate([{opacity: 1}, {opacity: 0}], {
      duration: reduced.matches ? 0 : 650, easing: 'ease-in-out', fill: 'forwards',
    });
    const finishEarly = () => {if (reduced.matches) animation.finish();};
    reduced.addEventListener('change', finishEarly);
    void animation.finished.then(() => {
      if (!canceled) setState({url, phase: 'completion'});
    }).catch(() => {/* Cancellation on navigation is expected. */});
    return () => {canceled = true; reduced.removeEventListener('change', finishEarly); animation.cancel();};
  }, [url, phase]);

  return {image, overlay, phase, decoded: phase === 'revealing' || phase === 'completion'};
}

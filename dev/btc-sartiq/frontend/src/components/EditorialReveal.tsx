import {useEffect, useRef, useState} from 'react';
import demoOriginal from '../assets/demo-original.jpg';
import demoResult from '../assets/demo-result.jpg';

export function EditorialReveal({compact = false}: {compact?: boolean}) {
  const element = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState(0);
  const [started, setStarted] = useState(false);
  const [complete, setComplete] = useState(false);

  useEffect(() => {
    const target = element.current;
    if (!target) return;
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduceMotion) { setPosition(50); setComplete(true); return; }
    const observer = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting || started) return;
      setStarted(true);
      requestAnimationFrame(() => setPosition(100));
      window.setTimeout(() => setComplete(true), 1900);
      observer.disconnect();
    }, {threshold: .35});
    observer.observe(target);
    return () => observer.disconnect();
  }, [started]);

  return <section ref={element} className={`editorial-reveal ${compact ? 'editorial-reveal--compact' : ''} ${started ? 'editorial-reveal--started' : ''}`} aria-label="Original and finished sartiq product image">
    <img src={demoOriginal} alt="Original striped garment photograph" />
    <div className="editorial-reveal__result" style={{clipPath: `inset(0 ${100 - position}% 0 0)`}}><img src={demoResult} alt="Finished studio product photograph" /></div>
    {!complete && <span className="editorial-reveal__line" style={{left: `${position}%`}} aria-hidden="true" />}
    <span className="editorial-reveal__label editorial-reveal__label--before">Before</span><span className="editorial-reveal__label editorial-reveal__label--after">After</span>
    <input aria-label="Move the before and after divider" type="range" min="0" max="100" value={position} onChange={event => { setComplete(true); setPosition(Number(event.target.value)); }}/>
  </section>;
}

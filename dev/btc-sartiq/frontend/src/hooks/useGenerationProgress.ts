import {useEffect, useRef, useState} from 'react';

import {estimateGenerationProgress} from '../lib/generationProgress';

export function useGenerationProgress(status: string, active: boolean, decoded: boolean, run: number | null) {
  const [percent, setPercent] = useState(0);
  const value = useRef(0);
  const stage = useRef({status, since: performance.now()});
  useEffect(() => {stage.current = {status, since: performance.now()};}, [status, run]);
  useEffect(() => {value.current = 0; setPercent(0);}, [run]);
  useEffect(() => {
    if (decoded && !active) {value.current = 100; setPercent(100); return;}
    if (!active || run === null) return;
    let frame = 0;
    let previous = performance.now();
    const tick = (now: number) => {
      const seconds = Math.min((now - previous) / 1000, .1);
      previous = now;
      const age = Math.max(0, (now - stage.current.since) / 1000);
      const elapsed = Math.max(0, (Date.now() - run) / 1000);
      const target = decoded ? 100 : estimateGenerationProgress(elapsed, stage.current.status, age);
      const gap = Math.max(0, target - value.current);
      const increment = decoded ? gap * (1 - Math.exp(-18 * seconds))
        : elapsed <= 40 ? gap : Math.min(gap, seconds * 1.2);
      value.current = Math.min(decoded ? 100 : 99, value.current + increment);
      setPercent(Math.min(decoded ? 100 : 99, Math.round(value.current)));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [active, decoded, run]);
  return percent;
}

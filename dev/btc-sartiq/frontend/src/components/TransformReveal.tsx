import {useEffect, useState} from 'react';

type Props = {original: string; result: string; onComplete: () => void};

export function TransformReveal({original, result, onComplete}: Props) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const frame = requestAnimationFrame(() => setProgress(100));
    const complete = window.setTimeout(onComplete, 1750);
    return () => { cancelAnimationFrame(frame); window.clearTimeout(complete); };
  }, [onComplete]);

  return <div className="transform-reveal" aria-label="Your image is transforming">
    <img src={original} alt="Original garment" />
    <div className="transform-reveal__result" style={{clipPath: `inset(0 ${100 - progress}% 0 0)`}}>
      <img src={result} alt="Professional product image" />
    </div>
    <div className="transform-reveal__scanner" style={{left: `${progress}%`}} aria-hidden="true" />
  </div>;
}

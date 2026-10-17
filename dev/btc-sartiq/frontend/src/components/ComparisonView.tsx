import {useState} from 'react';
import {ArrowLeftRight, X} from 'lucide-react';
import {GlassButton} from './GlassButton';

type Props = {original: string; result: string; onClose: () => void};

export function ComparisonView({original, result, onClose}: Props) {
  const [position, setPosition] = useState(50);
  return <section className="comparison-screen" aria-label="Compare original and result">
    <header className="comparison-screen__top"><GlassButton tone="dark" onClick={onClose} aria-label="Close comparison"><X size={20}/></GlassButton><span>Compare</span><span /></header>
    <div className="comparison-slider">
      <img src={original} alt="Original garment" />
      <div className="comparison-slider__result" style={{clipPath: `inset(0 0 0 ${position}%)`}}><img src={result} alt="Professional product image" /></div>
      <span className="comparison-slider__label comparison-slider__label--original">Original</span>
      <span className="comparison-slider__label comparison-slider__label--result">Result</span>
      <span className="comparison-slider__line" style={{left: `${position}%`}} aria-hidden="true"><ArrowLeftRight size={15}/></span>
      <input aria-label="Comparison position" type="range" min="0" max="100" value={position} onChange={event => setPosition(Number(event.target.value))}/>
    </div>
    <p>Drag to compare the original image with the finished product photograph.</p>
  </section>;
}

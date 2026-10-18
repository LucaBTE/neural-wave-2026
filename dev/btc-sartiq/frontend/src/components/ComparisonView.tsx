import {useState} from 'react';
import {ArrowLeftRight} from 'lucide-react';

type Props = {original: string; result: string};

export function ComparisonView({original, result}: Props) {
  const [position, setPosition] = useState(50);
  return <div className="comparison-slider" aria-label="Compare original and result">
    <img src={original} alt="Original garment" />
    <div className="comparison-slider__result" style={{clipPath: `inset(0 0 0 ${position}%)`}}><img src={result} alt="Professional product image" /></div>
    <span className="comparison-slider__label comparison-slider__label--original">Original</span>
    <span className="comparison-slider__label comparison-slider__label--result">Result</span>
    <span className="comparison-slider__line" style={{left: `${position}%`}} aria-hidden="true"><ArrowLeftRight size={15}/></span>
    <input aria-label="Comparison position" aria-valuetext={`${position}% original photo`} type="range" min="0" max="100" value={position} onChange={event => setPosition(Number(event.target.value))}/>
  </div>;
}

import {useState} from 'react';
import {ArrowLeftRight} from 'lucide-react';
import demoOriginal from '../assets/demo-original.jpg';
import demoResult from '../assets/demo-result.jpg';

export function EditorialReveal({compact = false}: {compact?: boolean}) {
  const [position, setPosition] = useState(50);
  return <section className={`editorial-reveal ${compact ? 'editorial-reveal--compact' : ''}`} aria-label="Original and finished sartiq product image">
    <img src={demoOriginal} alt="Original striped garment photograph" />
    <div className="editorial-reveal__result" style={{clipPath: `inset(0 0 0 ${position}%)`}}><img src={demoResult} alt="Finished studio product photograph" /></div>
    <span className="editorial-reveal__line" style={{left: `${position}%`}} aria-hidden="true"><ArrowLeftRight size={18}/></span>
    <span className="editorial-reveal__label editorial-reveal__label--before">Before</span><span className="editorial-reveal__label editorial-reveal__label--after">After</span>
    <input aria-label="Move the before and after divider" aria-valuetext={`${position}% original photo`} type="range" min="0" max="100" value={position} onChange={event => setPosition(Number(event.target.value))}/>
  </section>;
}

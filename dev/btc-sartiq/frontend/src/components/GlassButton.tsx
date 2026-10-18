import type {ButtonHTMLAttributes, ReactNode} from 'react';

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {children: ReactNode; tone?: 'light' | 'dark' | 'quiet'};

export function GlassButton({children, tone = 'light', className = '', ...props}: Props) {
  return <button {...props} className={`glass-button glass-button--${tone} ${className}`}>{children}</button>;
}

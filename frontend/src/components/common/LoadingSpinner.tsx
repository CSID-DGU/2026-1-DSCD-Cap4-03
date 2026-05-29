import './LoadingSpinner.css';

interface Props {
  text?: string;
  fullPage?: boolean;
}

export default function LoadingSpinner({ text = '불러오는 중이에요', fullPage = true }: Props) {
  return (
    <div className={`ls-wrap${fullPage ? ' ls-full' : ''}`}>
      <div className="ls-ring" />
      {text && <p className="ls-text">{text}</p>}
    </div>
  );
}

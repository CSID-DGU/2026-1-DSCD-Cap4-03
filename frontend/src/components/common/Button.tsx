type Props = {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary';
};

export default function Button({ children, onClick, variant = 'primary' }: Props) {
  return (
    <button className={`btn-${variant}`} onClick={onClick}>
      {children}
    </button>
  );
}
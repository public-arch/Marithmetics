export default function MathBlock({ children, label }) {
  return (
    <div style={{
      background: 'rgba(0,0,0,0.3)',
      border: '1px solid rgba(228,187,124,0.2)',
      borderRadius: 4,
      padding: '20px 24px',
      margin: '16px 0',
      fontFamily: 'var(--font-mono)', fontSize: 16,
      color: 'var(--gold)', textAlign: 'center',
      position: 'relative'
    }}>
      {label && (
        <div style={{
          position: 'absolute', top: -10, left: 16,
          background: 'var(--bg)', padding: '0 8px',
          fontSize: 11, color: 'var(--text-muted)',
          textTransform: 'uppercase', letterSpacing: '0.1em'
        }}>{label}</div>
      )}
      {children}
    </div>
  );
}

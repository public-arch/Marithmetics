export default function Card({ children, accent = false, borderLeft = false, style = {} }) {
  return (
    <div style={{
      background: accent ? 'var(--card-accent-bg)' : 'var(--card-bg)',
      border: `1px solid ${accent ? 'var(--border-accent)' : 'var(--border)'}`,
      borderLeft: borderLeft ? '4px solid var(--gold)' : undefined,
      borderRadius: 4,
      padding: 32,
      transition: 'all 0.3s ease',
      ...style
    }}>{children}</div>
  );
}

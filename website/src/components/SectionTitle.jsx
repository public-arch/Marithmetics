export default function SectionTitle({ title, subtitle }) {
  return (
    <div style={{ marginBottom: 32 }}>
      <h2 style={{
        fontFamily: 'var(--font-serif)', fontSize: 36,
        fontWeight: 400, color: 'var(--text)', margin: 0, marginBottom: 8
      }}>{title}</h2>
      {subtitle && (
        <p style={{
          fontFamily: 'var(--font-body)', fontSize: 16,
          color: 'var(--gold)', margin: 0, opacity: 0.7
        }}>{subtitle}</p>
      )}
    </div>
  );
}

export default function Footer() {
  return (
    <footer style={{
      background: 'var(--bg)',
      borderTop: '1px solid var(--border)',
      padding: '64px 24px',
      textAlign: 'center',
      marginTop: 'auto',
      position: 'relative'
    }}>
      {/* Gold divider line above footer */}
      <div style={{
        height: '2px',
        background: 'linear-gradient(90deg, transparent, var(--gold), transparent)',
        marginBottom: 48,
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0
      }} />

      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        <div style={{
          fontFamily: 'var(--font-serif)',
          fontSize: 22,
          color: 'var(--gold)',
          marginBottom: 20,
          letterSpacing: '0.05em'
        }}>
          Marithmetics
        </div>
        <p style={{
          fontFamily: 'var(--font-body)',
          fontSize: 14,
          color: 'var(--text-muted)',
          margin: '0 0 16px 0',
          lineHeight: 1.8
        }}>
          A deterministic, audit-grade pipeline for integer-to-physics emergence.
          <br/>
          <a href="https://github.com/public-arch/Marithmetics" target="_blank" rel="noopener noreferrer"
            style={{
              color: 'var(--gold)',
              textDecoration: 'none',
              transition: 'color 0.2s ease'
            }}
            onMouseEnter={(e) => e.target.style.color = '#f0d99c'}
            onMouseLeave={(e) => e.target.style.color = 'var(--gold)'}
          >
            GitHub Repository
          </a>
          {' · '}
          MIT License
        </p>

        {/* Subtle watermark */}
        <p style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          color: 'rgba(228, 187, 124, 0.2)',
          margin: '24px 0 0 0',
          letterSpacing: '0.1em',
          fontWeight: 300
        }}>
          137 · 107 · 103
        </p>
      </div>
    </footer>
  );
}

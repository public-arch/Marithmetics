import { Link, useLocation } from 'react-router-dom';

const NAV_ITEMS = [
  { path: '/', label: 'Home' },
  { path: '/discovery', label: 'Discovery' },
  { path: '/demos', label: 'Demos' },
  { path: '/papers', label: 'Papers' },
  { path: '/falsification', label: 'Falsification' },
  { path: '/methodology', label: 'Methodology' },
  { path: '/about', label: 'About' }
];

export default function SiteHeader() {
  const location = useLocation();
  return (
    <header style={{
      background: 'rgba(13,17,23,0.95)',
      backdropFilter: 'blur(12px)',
      borderBottom: '1px solid var(--border)',
      position: 'sticky',
      top: 0,
      zIndex: 100,
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.4)'
    }}>
      {/* Top gold accent line */}
      <div style={{
        height: '2px',
        background: 'var(--gold)',
        width: '100%'
      }} />
      
      <div style={{
        maxWidth: 1200,
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        gap: 0
      }}>
        <Link to="/" style={{
          fontFamily: 'var(--font-serif)',
          fontSize: 24,
          color: 'var(--gold)',
          textDecoration: 'none',
          padding: '18px 32px',
          fontWeight: 600,
          letterSpacing: '0.05em',
          transition: 'color 0.2s ease'
        }}
        onMouseEnter={(e) => e.target.style.color = '#f0d99c'}
        onMouseLeave={(e) => e.target.style.color = 'var(--gold)'}
        >
          Marithmetics
        </Link>
        <nav style={{ display: 'flex', gap: 0, overflow: 'auto', flex: 1 }}>
          {NAV_ITEMS.map(item => (
            <Link
              key={item.path}
              to={item.path}
              style={{
                background: location.pathname === item.path ? 'rgba(228,187,124,0.08)' : 'transparent',
                borderBottom: location.pathname === item.path ? '3px solid var(--gold)' : '3px solid transparent',
                padding: '18px 20px',
                fontFamily: 'var(--font-body)',
                fontSize: 14,
                color: location.pathname === item.path ? 'var(--gold)' : 'var(--text-muted)',
                textDecoration: 'none',
                whiteSpace: 'nowrap',
                transition: 'all 0.2s ease',
                letterSpacing: '0.3px'
              }}
            >{item.label}</Link>
          ))}
        </nav>
      </div>
    </header>
  );
}

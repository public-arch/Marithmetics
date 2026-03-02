import { Link } from 'react-router-dom';
import Card from '../components/Card';
import Badge from '../components/Badge';
import SectionTitle from '../components/SectionTitle';
import demos from '../data/demos.json';
import { CATEGORY_COLORS } from '../utils/categories';

const FEATURED_DEMO_IDS = ['demo-34', 'demo-36', 'demo-37', 'demo-70'];

export default function HomePage() {
  const featured = demos.filter(d => FEATURED_DEMO_IDS.includes(d.id));

  return (
    <div style={{ flex: 1 }}>

      {/* ════════ HERO ════════ */}
      <section style={{
        background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
        position: 'relative',
        overflow: 'hidden',
        padding: '140px 48px 100px',
      }}>
        {/* Ambient glow overlays */}
        <div style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          background: `radial-gradient(ellipse 80% 60% at 20% 80%, rgba(228,187,124,0.06) 0%, transparent 60%),
                       radial-gradient(ellipse 60% 80% at 85% 20%, rgba(15,52,96,0.35) 0%, transparent 60%)`
        }} />

        {/* Triangle SVG — behind content */}
        <svg viewBox="0 0 400 360" style={{
          position: 'absolute', right: '4%', top: '50%', transform: 'translateY(-50%)',
          width: 560, height: 500, opacity: 0.25, pointerEvents: 'none',
          filter: 'drop-shadow(0 0 80px rgba(228,187,124,0.18))'
        }}>
          <defs>
            <radialGradient id="glow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="rgba(228,187,124,0.3)" />
              <stop offset="100%" stopColor="rgba(228,187,124,0)" />
            </radialGradient>
            <radialGradient id="glowPulse" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="rgba(228,187,124,0.45)" />
              <stop offset="60%" stopColor="rgba(228,187,124,0.12)" />
              <stop offset="100%" stopColor="rgba(228,187,124,0)" />
            </radialGradient>
            <linearGradient id="triFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(228,187,124,0.08)" />
              <stop offset="100%" stopColor="rgba(228,187,124,0.02)" />
            </linearGradient>
          </defs>
          <circle cx="200" cy="180" r="180" fill="url(#glow)" />
          {/* Breathing pulse layer */}
          <circle cx="200" cy="180" r="200" fill="url(#glowPulse)" opacity="0">
            <animate attributeName="opacity" values="0;0.6;0" dur="7s" repeatCount="indefinite" calcMode="spline" keySplines="0.45 0 0.55 1;0.45 0 0.55 1" />
          </circle>
          <polygon points="200,30 350,280 50,280" fill="url(#triFill)" stroke="#e4bb7c" strokeWidth="2.5" />
          <text x="200" y="18" textAnchor="middle" fill="#e4bb7c" fontSize="22" fontWeight="bold" fontFamily="'JetBrains Mono', monospace">137</text>
          <text x="358" y="298" textAnchor="start" fill="#e4bb7c" fontSize="22" fontWeight="bold" fontFamily="'JetBrains Mono', monospace">107</text>
          <text x="42" y="298" textAnchor="end" fill="#e4bb7c" fontSize="22" fontWeight="bold" fontFamily="'JetBrains Mono', monospace">103</text>
          <circle cx="200" cy="180" r="5" fill="#e4bb7c" />
          <circle cx="200" cy="180" r="9" fill="none" stroke="#e4bb7c" strokeWidth="1.5" opacity="0.5" />
          <line x1="200" y1="30" x2="200" y2="180" stroke="#e4bb7c" strokeWidth="1.5" strokeDasharray="5" opacity="0.6" />
          <line x1="350" y1="280" x2="200" y2="180" stroke="#e4bb7c" strokeWidth="1.5" strokeDasharray="5" opacity="0.6" />
          <line x1="50" y1="280" x2="200" y2="180" stroke="#e4bb7c" strokeWidth="1.5" strokeDasharray="5" opacity="0.6" />
        </svg>

        <div style={{ maxWidth: 1200, margin: '0 auto', position: 'relative', zIndex: 1 }}>
          {/* Eyebrow */}
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--gold)',
            letterSpacing: '0.25em', textTransform: 'uppercase', marginBottom: 28, fontWeight: 500
          }}>
            Deterministic Integer-to-Physics Emergence
          </div>

          {/* Title */}
          <h1 style={{
            fontFamily: 'var(--font-serif)', fontSize: 'clamp(56px, 9vw, 96px)',
            fontWeight: 300, color: 'var(--text)', margin: '0 0 24px',
            letterSpacing: '-2px', lineHeight: 1.05
          }}>
            Marithmetics
          </h1>

          {/* Subtitle */}
          <p style={{
            fontFamily: 'var(--font-body)', fontSize: 'clamp(19px, 2.4vw, 26px)',
            color: 'var(--text-muted)', margin: '0 0 64px', maxWidth: 680,
            lineHeight: 1.75, fontWeight: 300
          }}>
            Deriving the fine-structure constant, particle mass ratios, and Standard Model structure from pure integer geometry — zero free parameters, zero curve-fitting, fully falsifiable.
          </p>

          {/* Stats */}
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
            borderTop: '1px solid rgba(228,187,124,0.2)',
            borderBottom: '1px solid rgba(228,187,124,0.2)',
            padding: '44px 0', marginBottom: 56, maxWidth: 820
          }}>
            {[
              ['0', 'Free Parameters'],
              ['29', 'Deterministic Demos'],
              ['16', 'Papers'],
              ['3', 'Attack Vectors']
            ].map(([num, label], i) => (
              <div key={i} style={{
                textAlign: 'center',
                borderRight: i < 3 ? '1px solid rgba(228,187,124,0.2)' : 'none',
                padding: '0 16px'
              }}>
                <div style={{
                  fontFamily: 'var(--font-serif)', fontSize: 52, fontWeight: 300,
                  color: 'var(--gold)', marginBottom: 10, letterSpacing: '1px', lineHeight: 1
                }}>{num}</div>
                <div style={{
                  fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)',
                  letterSpacing: '0.12em', textTransform: 'uppercase'
                }}>{label}</div>
              </div>
            ))}
          </div>

          {/* CTA Buttons */}
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
            <Link to="/demos" style={{ textDecoration: 'none' }}>
              <button style={{
                fontFamily: 'var(--font-serif)', fontSize: 17, padding: '18px 48px',
                backgroundColor: 'var(--gold)', color: 'var(--bg)', border: 'none',
                borderRadius: 4, cursor: 'pointer', fontWeight: 500,
                transition: 'all 0.3s ease', letterSpacing: '0.5px'
              }}
              onMouseEnter={e => { e.target.style.boxShadow = '0 8px 32px rgba(228,187,124,0.35)'; e.target.style.transform = 'translateY(-2px)'; }}
              onMouseLeave={e => { e.target.style.boxShadow = 'none'; e.target.style.transform = 'translateY(0)'; }}
              >Explore Demos</button>
            </Link>
            <Link to="/falsification" style={{ textDecoration: 'none' }}>
              <button style={{
                fontFamily: 'var(--font-serif)', fontSize: 17, padding: '18px 48px',
                backgroundColor: 'transparent', color: 'var(--gold)',
                border: '2px solid var(--gold)', borderRadius: 4, cursor: 'pointer',
                fontWeight: 500, transition: 'all 0.3s ease', letterSpacing: '0.5px'
              }}
              onMouseEnter={e => { e.target.style.backgroundColor = 'rgba(228,187,124,0.08)'; e.target.style.transform = 'translateY(-2px)'; }}
              onMouseLeave={e => { e.target.style.backgroundColor = 'transparent'; e.target.style.transform = 'translateY(0)'; }}
              >How to Break This</button>
            </Link>
          </div>
        </div>
      </section>

      {/* ════════ BODY CONTENT ════════ */}
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 48px' }}>

        {/* ── Falsification Cards ── */}
        <section style={{ margin: '100px 0' }}>
          <div style={{ height: 2, background: 'linear-gradient(90deg, transparent, var(--gold), transparent)', marginBottom: 64 }} />

          <SectionTitle
            title="How to Break This"
            subtitle="Falsification testers — three ways the theory could fail (and why it doesn't)"
          />

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 28, marginTop: 40 }}>
            {[
              { q: "It's just numerology", a: "Run DEMO-64 (base-gauge invariance). The same integer triple (137, 107, 103) emerges across bases 2, 7, 10, and 16 — not possible under random selection.", to: '/demos/demo-64-base-gauge-invariance-integer-selector' },
              { q: 'Parameters were tuned', a: 'Counterfactual triples fail under the same deterministic pipeline. Degradation exceeds 1 + ε. No free parameters. No knobs.', to: '/falsification' },
              { q: 'Operators are arbitrary', a: 'Illegal controls (sharp cutoff, signed kernels) violate admissibility contracts. The lawful operator outperforms both. Falsifiable by design.', to: '/methodology' }
            ].map((c, i) => (
              <Link key={i} to={c.to} style={{ textDecoration: 'none', color: 'inherit' }}>
                <Card borderLeft style={{ cursor: 'pointer', height: '100%' }}>
                  <h3 style={{ fontFamily: 'var(--font-serif)', fontSize: 19, fontWeight: 400, color: 'var(--text)', margin: '0 0 14px', fontStyle: 'italic' }}>
                    "{c.q}"
                  </h3>
                  <p style={{ fontSize: 15, color: 'var(--text-muted)', margin: '0 0 24px', lineHeight: 1.75 }}>
                    {c.a}
                  </p>
                  <span style={{ color: 'var(--gold)', fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 500 }}>
                    Learn more →
                  </span>
                </Card>
              </Link>
            ))}
          </div>

          <div style={{ marginTop: 48, textAlign: 'center' }}>
            <Link to="/falsification" style={{ fontFamily: 'var(--font-serif)', fontSize: 17, color: 'var(--gold)', fontWeight: 500 }}>
              Full falsification framework →
            </Link>
          </div>
        </section>

        {/* ── Featured Flagships ── */}
        <section style={{ margin: '100px 0' }}>
          <SectionTitle
            title="Featured Flagships"
            subtitle="Master-grade demos spanning foundations, standard model, cosmology, and beyond"
          />

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 28, marginTop: 40 }}>
            {featured.map(demo => (
              <Link key={demo.id} to={`/demos/${demo.slug}`} style={{ textDecoration: 'none', color: 'inherit' }}>
                <div
                  style={{
                    background: 'var(--card-bg)', border: '1px solid var(--border)',
                    borderTop: `3px solid ${CATEGORY_COLORS[demo.category] || 'var(--gold)'}66`,
                    borderRadius: 4, padding: 32, cursor: 'pointer',
                    transition: 'all 0.3s ease', height: '100%',
                    display: 'flex', flexDirection: 'column'
                  }}
                  onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = '0 0 24px rgba(228,187,124,0.15)'; }}
                  onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none'; }}
                >
                  <div style={{ marginBottom: 14 }}><Badge category={demo.category} /></div>
                  <h3 style={{ fontFamily: 'var(--font-serif)', fontSize: 21, fontWeight: 400, color: 'var(--text)', margin: '0 0 12px', lineHeight: 1.3 }}>
                    {demo.shortTitle}
                  </h3>
                  <p style={{ fontSize: 14, color: 'var(--text-muted)', margin: '0 0 20px', lineHeight: 1.7, flex: 1 }}>
                    {demo.description}
                  </p>
                  <div><Badge text="Flagship" ok /></div>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* ── Philosophy banner ── */}
        <section style={{
          background: 'linear-gradient(135deg, rgba(26,26,46,0.5) 0%, rgba(22,33,62,0.5) 50%, rgba(15,52,96,0.5) 100%)',
          borderLeft: '4px solid var(--gold)', padding: '72px 64px', margin: '100px 0'
        }}>
          <h2 style={{
            fontFamily: 'var(--font-serif)', fontSize: 'clamp(38px, 5vw, 60px)',
            fontWeight: 300, color: 'var(--gold)', margin: '0 0 56px',
            lineHeight: 1.15, fontStyle: 'italic', letterSpacing: '-0.5px'
          }}>
            Narrative is not evidence.<br />Execution is.
          </h2>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 48 }}>
            {[
              ['Runnable Claims', 'Every physical prediction is backed by a deterministic, runnable demo. No hand-waving. No tuned parameters. Code is law.'],
              ['Authority Records', 'All results sealed in a cryptographic Authority-of-Record bundle. Timestamps. SHA-256 hashes. Full auditability.'],
              ['Falsifiable Design', 'Three explicit attack vectors. Three independent admissibility proofs. Counterfactual triples. Legal or illegal. No ambiguity.']
            ].map(([title, body], i) => (
              <div key={i}>
                <h3 style={{ fontFamily: 'var(--font-serif)', fontSize: 22, fontWeight: 400, color: 'var(--text)', margin: '0 0 14px' }}>{title}</h3>
                <p style={{ fontSize: 15, color: 'var(--text-muted)', margin: 0, lineHeight: 1.85 }}>{body}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Final CTA ── */}
        <section style={{ textAlign: 'center', margin: '100px 0 80px' }}>
          <div style={{ height: 2, background: 'linear-gradient(90deg, transparent, var(--gold), transparent)', marginBottom: 64 }} />
          <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: 'clamp(34px, 5vw, 52px)', fontWeight: 300, color: 'var(--text)', margin: '0 0 20px', letterSpacing: '-0.5px' }}>
            Ready to explore?
          </h2>
          <p style={{ fontSize: 18, color: 'var(--text-muted)', margin: '0 auto 44px', maxWidth: 640, lineHeight: 1.8, fontWeight: 300 }}>
            Start with a flagship demo, review the papers, or dive into the methodology.
          </p>
          <Link to="/demos" style={{ textDecoration: 'none' }}>
            <button style={{
              fontFamily: 'var(--font-serif)', fontSize: 17, padding: '18px 56px',
              backgroundColor: 'var(--gold)', color: 'var(--bg)', border: 'none',
              borderRadius: 4, cursor: 'pointer', fontWeight: 500,
              transition: 'all 0.3s ease', letterSpacing: '0.5px'
            }}
            onMouseEnter={e => { e.target.style.boxShadow = '0 8px 32px rgba(228,187,124,0.35)'; e.target.style.transform = 'translateY(-2px)'; }}
            onMouseLeave={e => { e.target.style.boxShadow = 'none'; e.target.style.transform = 'translateY(0)'; }}
            >View all 29 demos</button>
          </Link>
        </section>
      </div>
    </div>
  );
}

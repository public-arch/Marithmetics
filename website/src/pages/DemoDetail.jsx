import { useParams, Link } from 'react-router-dom';
import Card from '../components/Card';
import Badge from '../components/Badge';
import SectionTitle from '../components/SectionTitle';
import MathBlock from '../components/MathBlock';
import AudioPlayer from '../components/AudioPlayer';
import demos from '../data/demos.json';
import { getCategoryLabel } from '../utils/categories';

export default function DemoDetail() {
  const { slug } = useParams();
  const demo = demos.find(d => d.slug === slug);

  if (!demo) {
    return (
      <div style={{ padding: '40px 24px', maxWidth: 1200, margin: '0 auto', flex: 1 }}>
        <SectionTitle title="Demo Not Found" subtitle="The demo you're looking for doesn't exist." />
        <Link
          to="/demos"
          style={{
            color: 'var(--gold)',
            textDecoration: 'none',
            fontSize: 14,
            fontWeight: 500,
            fontFamily: 'var(--font-mono)',
            display: 'inline-block',
            marginTop: 24
          }}
        >
          ← Back to Demos
        </Link>
      </div>
    );
  }

  // GitHub paths
  const REPO = 'https://github.com/public-arch/Marithmetics';
  const demoDir = `demos/${demo.category}/${demo.slug}`;
  const demoPath = `${demoDir}/demo.py`;
  const githubDemoDir = `${REPO}/tree/main/${demoDir}`;
  const githubDemoFile = `${REPO}/blob/main/${demoPath}`;

  return (
    <div style={{ padding: '40px 24px', maxWidth: 1200, margin: '0 auto', flex: 1 }}>
      {/* Back Link */}
      <Link
        to="/demos"
        style={{
          color: 'var(--text-muted)',
          textDecoration: 'none',
          fontSize: 12,
          fontWeight: 500,
          fontFamily: 'var(--font-mono)',
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          marginBottom: 48,
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
          opacity: 0.6,
          transition: 'opacity 0.2s, color 0.2s'
        }}
        onMouseEnter={e => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.color = 'var(--gold)'; }}
        onMouseLeave={e => { e.currentTarget.style.opacity = '0.6'; e.currentTarget.style.color = 'var(--text-muted)'; }}
      >
        ← All Demos
      </Link>

      {/* Header Section */}
      <div style={{ marginBottom: 48 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
          <Badge text={demo.id} ok={true} />
          <Badge category={demo.category} />
          <Badge text={demo.status} ok={demo.status === 'certified'} />
          {demo.flagship && <Badge text="Flagship" ok={true} />}
          <a
            href={githubDemoDir}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              color: 'var(--text-muted)', textDecoration: 'none',
              fontFamily: 'var(--font-mono)', fontSize: 11, opacity: 0.5,
              transition: 'opacity 0.2s',
              marginLeft: 'auto'
            }}
            onMouseEnter={e => e.currentTarget.style.opacity = '1'}
            onMouseLeave={e => e.currentTarget.style.opacity = '0.5'}
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
            Source
          </a>
        </div>

        <h1 style={{
          fontFamily: 'var(--font-serif)',
          fontSize: 'clamp(32px, 5vw, 48px)',
          fontWeight: 400,
          color: 'var(--text)',
          margin: '0 0 8px 0',
          lineHeight: 1.2,
          letterSpacing: '-0.01em'
        }}>
          {demo.shortTitle}
        </h1>

        {demo.title !== demo.shortTitle && (
          <p style={{
            fontFamily: 'var(--font-serif)',
            fontSize: 18,
            fontWeight: 300,
            color: 'var(--text-muted)',
            margin: '0 0 0 0',
            lineHeight: 1.5,
            maxWidth: 800,
            fontStyle: 'italic'
          }}>
            {demo.title}
          </p>
        )}
      </div>

      {/* ─── Gold divider ─── */}
      <div style={{
        height: 1,
        background: 'linear-gradient(90deg, var(--gold), rgba(228,187,124,0.2) 60%, transparent)',
        marginBottom: 48
      }} />

      {/* Claim — the thesis statement */}
      {demo.claim && (
        <div style={{ marginBottom: 56 }}>
          <div style={{
            borderLeft: '3px solid var(--gold)',
            paddingLeft: 28,
            maxWidth: 800
          }}>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--gold)',
              textTransform: 'uppercase',
              letterSpacing: '0.12em',
              marginBottom: 12,
              fontWeight: 600
            }}>
              Claim
            </div>
            <p style={{
              fontFamily: 'var(--font-body)',
              fontSize: 17,
              color: 'var(--text)',
              margin: 0,
              lineHeight: 1.8,
              fontWeight: 400
            }}>
              {demo.claim}
            </p>
          </div>
        </div>
      )}

      {/* Result — evidence with metrics dashboard */}
      {demo.result && (
        <div style={{ marginBottom: 56 }}>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.12em',
            marginBottom: 16,
            fontWeight: 500
          }}>
            Result
          </div>
          <p style={{
            fontFamily: 'var(--font-body)',
            fontSize: 16,
            color: 'rgba(255,255,255,0.75)',
            margin: '0 0 28px 0',
            lineHeight: 1.8,
            maxWidth: 800
          }}>
            {demo.result}
          </p>

          {/* Key Outputs — instrument readout grid */}
          {demo.keyOutputs && demo.keyOutputs.length > 0 && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${Math.min(demo.keyOutputs.length, 4)}, 1fr)`,
              gap: 1,
              background: 'var(--border)',
              borderRadius: 6,
              overflow: 'hidden',
              boxShadow: '0 2px 16px rgba(0,0,0,0.3)'
            }}>
              {demo.keyOutputs.map((output, i) => (
                <div key={i} style={{
                  padding: '20px 24px',
                  background: 'var(--bg)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 4
                }}>
                  <div style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 10,
                    color: 'var(--text-muted)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.08em',
                    opacity: 0.7
                  }}>
                    {output.label}
                  </div>
                  <div style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 22,
                    fontWeight: 600,
                    color: 'var(--gold)',
                    lineHeight: 1.2,
                    textShadow: '0 0 24px rgba(228,187,124,0.15)'
                  }}>
                    {output.value}
                  </div>
                  {output.note && (
                    <div style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: 10,
                      color: 'var(--text-muted)',
                      opacity: 0.5
                    }}>
                      {output.note}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Falsification — the critical section */}
      {demo.falsification && (
        <div style={{ marginBottom: 56 }}>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.12em',
            marginBottom: 8,
            fontWeight: 500
          }}>
            Falsification Design
          </div>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--gold)',
            opacity: 0.5,
            marginBottom: 20,
            letterSpacing: '0.02em'
          }}>
            How to break this claim
          </div>
          <div style={{
            background: 'rgba(228,187,124,0.04)',
            border: '1px solid rgba(228,187,124,0.15)',
            borderLeft: '3px solid rgba(228,187,124,0.4)',
            borderRadius: '0 6px 6px 0',
            padding: '28px 32px'
          }}>
            <p style={{
              fontFamily: 'var(--font-body)',
              fontSize: 15,
              color: 'var(--text)',
              margin: 0,
              lineHeight: 1.8
            }}>
              {demo.falsification}
            </p>
          </div>
        </div>
      )}

      {/* Fallback description if no structured fields */}
      {!demo.claim && (
        <div style={{ marginBottom: 40 }}>
          <p style={{
            fontSize: 16,
            color: 'var(--text-muted)',
            margin: 0,
            lineHeight: 1.7,
            maxWidth: 800
          }}>
            {demo.description}
          </p>
        </div>
      )}

      {/* ─── Thin divider ─── */}
      <div style={{
        height: 1,
        background: 'linear-gradient(90deg, var(--border), transparent 80%)',
        marginBottom: 48
      }} />

      {/* Audio Overview Section */}
      <div style={{ marginBottom: 56 }}>
        <AudioPlayer
          src={`/audio/demo-${demo.id.split('-')[1]}.m4a`}
          title="Audio Overview"
          demoId={demo.id}
        />
      </div>

      {/* How to Reproduce Section */}
      <div style={{ marginBottom: 56 }}>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          color: 'var(--text-muted)',
          textTransform: 'uppercase',
          letterSpacing: '0.12em',
          marginBottom: 8,
          fontWeight: 500
        }}>
          Reproduce
        </div>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          color: 'var(--gold)',
          opacity: 0.5,
          marginBottom: 20,
          letterSpacing: '0.02em'
        }}>
          Run the demo locally from source
        </div>
        <MathBlock label="Command">
          python {demoPath}
        </MathBlock>
        <div style={{ display: 'flex', gap: 16, marginTop: 24, flexWrap: 'wrap', alignItems: 'center' }}>
          <a
            href={githubDemoFile}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 8,
              padding: '10px 20px', borderRadius: 4,
              border: '1px solid rgba(228,187,124,0.3)',
              color: 'var(--gold)', textDecoration: 'none',
              fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 500,
              transition: 'all 0.2s'
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--gold)'; e.currentTarget.style.background = 'rgba(228,187,124,0.06)'; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(228,187,124,0.3)'; e.currentTarget.style.background = 'transparent'; }}
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
            View demo.py
          </a>
          <a
            href={githubDemoDir}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 8,
              padding: '10px 20px', borderRadius: 4,
              border: '1px solid rgba(228,187,124,0.12)',
              color: 'var(--text-muted)', textDecoration: 'none',
              fontFamily: 'var(--font-mono)', fontSize: 13,
              transition: 'all 0.2s'
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(228,187,124,0.3)'; e.currentTarget.style.color = 'var(--gold)'; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(228,187,124,0.12)'; e.currentTarget.style.color = 'var(--text-muted)'; }}
          >
            Browse full folder →
          </a>
          <span style={{
            fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', opacity: 0.4,
            letterSpacing: '0.04em'
          }}>
            stdlib-only · deterministic · reproducible
          </span>
        </div>
      </div>

      {/* Visualizer Note */}
      {demo.hasVisualizer && (
        <div style={{ marginBottom: 56 }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 14,
            padding: '20px 28px',
            background: 'rgba(228,187,124,0.04)',
            border: '1px solid rgba(228,187,124,0.12)',
            borderRadius: 6
          }}>
            <span style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 28,
              height: 28,
              borderRadius: '50%',
              background: 'rgba(228,187,124,0.12)',
              color: 'var(--gold)',
              fontSize: 13,
              flexShrink: 0
            }}>
              ⚡
            </span>
            <div>
              <div style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 13,
                fontWeight: 500,
                color: 'var(--text)',
                marginBottom: 2
              }}>
                Interactive Visualizer Available
              </div>
              <div style={{
                fontFamily: 'var(--font-body)',
                fontSize: 13,
                color: 'var(--text-muted)',
                opacity: 0.7
              }}>
                Run the demo locally for interactive output
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Related Demos */}
      <div style={{
        marginTop: 64,
        paddingTop: 40,
        borderTop: '1px solid var(--border)'
      }}>
        <Link
          to="/demos"
          style={{
            color: 'var(--gold)',
            textDecoration: 'none',
            fontSize: 13,
            fontWeight: 500,
            fontFamily: 'var(--font-mono)',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
            opacity: 0.8,
            transition: 'opacity 0.2s'
          }}
          onMouseEnter={e => e.currentTarget.style.opacity = '1'}
          onMouseLeave={e => e.currentTarget.style.opacity = '0.8'}
        >
          View all demos →
        </Link>
      </div>
    </div>
  );
}

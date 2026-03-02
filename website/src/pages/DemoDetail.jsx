import { useParams, Link } from 'react-router-dom';
import Card from '../components/Card';
import Badge from '../components/Badge';
import SectionTitle from '../components/SectionTitle';
import MathBlock from '../components/MathBlock';
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
          color: 'var(--gold)',
          textDecoration: 'none',
          fontSize: 13,
          fontWeight: 500,
          fontFamily: 'var(--font-mono)',
          display: 'inline-block',
          marginBottom: 32
        }}
      >
        ← Back to Demos
      </Link>

      {/* Header Section */}
      <div style={{ marginBottom: 40 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
          <Badge text={demo.id} ok={true} />
          <a
            href={githubDemoDir}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              color: 'var(--text-muted)', textDecoration: 'none',
              fontFamily: 'var(--font-mono)', fontSize: 12, opacity: 0.7,
              transition: 'opacity 0.2s'
            }}
            onMouseEnter={e => e.currentTarget.style.opacity = '1'}
            onMouseLeave={e => e.currentTarget.style.opacity = '0.7'}
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
            View source
          </a>
        </div>

        <h1 style={{
          fontFamily: 'var(--font-serif)',
          fontSize: 'clamp(28px, 5vw, 42px)',
          fontWeight: 400,
          color: 'var(--text)',
          margin: '0 0 8px 0',
          lineHeight: 1.3
        }}>
          {demo.shortTitle}
        </h1>

        {demo.title !== demo.shortTitle && (
          <p style={{
            fontFamily: 'var(--font-serif)',
            fontSize: 16,
            fontWeight: 300,
            color: 'var(--text-muted)',
            margin: '0 0 16px 0',
            lineHeight: 1.5,
            maxWidth: 800,
            fontStyle: 'italic'
          }}>
            {demo.title}
          </p>
        )}

        <div style={{ display: 'flex', gap: 12, marginTop: 16, flexWrap: 'wrap' }}>
          <Badge category={demo.category} />
          <Badge text={demo.status} ok={demo.status === 'certified'} />
          {demo.flagship && <Badge text="Flagship" ok={true} />}
        </div>
      </div>

      {/* Description */}
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

      {/* Falsification Gates Section */}
      <div style={{ marginBottom: 48 }}>
        <SectionTitle
          title="Falsification Gates"
          subtitle="Built-in tests that provide negative controls and counterfactual validation"
        />
        <Card accent={true}>
          <p style={{
            fontSize: 15,
            color: 'var(--text)',
            margin: '0 0 12px 0',
            lineHeight: 1.6
          }}>
            Each demo includes falsification gates—built-in negative controls and counterfactual tests that can break the claim.
          </p>
          <p style={{
            fontSize: 15,
            color: 'var(--text-muted)',
            margin: 0,
            lineHeight: 1.6
          }}>
            These aren't tests the author hoped would pass. They're tests designed to fail if the core claim is false. Counterfactual triples, illegal operators, and base-variant checks all run through the same deterministic pipeline. If the method is brittle or fit-dependent, the falsification gates will catch it.
          </p>
        </Card>
      </div>

      {/* How to Reproduce Section */}
      <div style={{ marginBottom: 48 }}>
        <SectionTitle
          title="How to Reproduce"
          subtitle="Run the demo locally from source"
        />
        <MathBlock label="Command">
          python {demoPath}
        </MathBlock>
        <div style={{ display: 'flex', gap: 16, marginTop: 20, flexWrap: 'wrap', alignItems: 'center' }}>
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
              border: '1px solid rgba(228,187,124,0.15)',
              color: 'var(--text-muted)', textDecoration: 'none',
              fontFamily: 'var(--font-mono)', fontSize: 13,
              transition: 'all 0.2s'
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(228,187,124,0.3)'; e.currentTarget.style.color = 'var(--gold)'; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(228,187,124,0.15)'; e.currentTarget.style.color = 'var(--text-muted)'; }}
          >
            Browse full folder →
          </a>
          <span style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', opacity: 0.6 }}>
            stdlib-only · deterministic · reproducible
          </span>
        </div>
      </div>

      {/* Visualizer Note */}
      {demo.hasVisualizer && (
        <div style={{ marginBottom: 48 }}>
          <Card accent={true}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
              <span style={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 24,
                height: 24,
                borderRadius: '50%',
                background: 'rgba(228,187,124,0.15)',
                color: 'var(--gold)',
                fontWeight: 'bold',
                fontSize: 14
              }}>
                ⚡
              </span>
              <h3 style={{
                fontFamily: 'var(--font-serif)',
                fontSize: 18,
                fontWeight: 400,
                color: 'var(--text)',
                margin: 0
              }}>
                Interactive Visualizer Available
              </h3>
            </div>
            <p style={{
              fontSize: 15,
              color: 'var(--text-muted)',
              margin: 0,
              lineHeight: 1.6
            }}>
              This demo includes an interactive visualizer. Check the output when you run the demo for instructions.
            </p>
          </Card>
        </div>
      )}

      {/* Related Demos */}
      <div style={{ marginTop: 60, paddingTop: 40, borderTop: '1px solid var(--border)' }}>
        <SectionTitle title="More Demos" subtitle="Explore other demonstrations" />
        <Link
          to="/demos"
          style={{
            color: 'var(--gold)',
            textDecoration: 'none',
            fontSize: 14,
            fontWeight: 500,
            fontFamily: 'var(--font-mono)',
            display: 'inline-block'
          }}
        >
          View all demos →
        </Link>
      </div>
    </div>
  );
}

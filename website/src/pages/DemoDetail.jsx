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

  // Extract category slug from demo path (e.g., demo-64-... -> substrate)
  const demoPath = `demos/${demo.category}/${demo.slug}/demo.py`;

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
        <div style={{ marginBottom: 16 }}>
          <Badge text={demo.id} ok={true} />
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
        <p style={{
          fontSize: 13,
          color: 'var(--text-muted)',
          margin: '16px 0 0 0',
          fontFamily: 'var(--font-mono)'
        }}>
          All demos are stdlib-only or minimal-dependency and deterministic.
        </p>
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

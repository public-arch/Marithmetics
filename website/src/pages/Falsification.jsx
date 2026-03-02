import { Link } from 'react-router-dom';
import Card from '../components/Card';
import Badge from '../components/Badge';
import SectionTitle from '../components/SectionTitle';
import MathBlock from '../components/MathBlock';
import demos from '../data/demos.json';

const AttackDefenseCard = ({ attack, defense, demoLinks, linkLabel }) => (
  <Card accent={false} style={{ marginTop: 24 }}>
    {/* Attack Section */}
    <div style={{ marginBottom: 24, paddingBottom: 24, borderBottom: '1px solid rgba(248,113,113,0.2)' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        marginBottom: 12
      }}>
        <span style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 24,
          height: 24,
          borderRadius: '50%',
          background: 'rgba(248,113,113,0.15)',
          color: '#f87171',
          fontWeight: 'bold',
          fontSize: 14
        }}>
          ✗
        </span>
        <h4 style={{
          fontFamily: 'var(--font-serif)',
          fontSize: 16,
          fontWeight: 400,
          color: '#f87171',
          margin: 0
        }}>
          The Attack
        </h4>
      </div>
      <p style={{
        fontSize: 15,
        color: 'var(--text)',
        margin: 0,
        lineHeight: 1.6
      }}>
        {attack}
      </p>
    </div>

    {/* Defense Section */}
    <div>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        marginBottom: 12
      }}>
        <span style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 24,
          height: 24,
          borderRadius: '50%',
          background: 'rgba(74,222,128,0.15)',
          color: '#4ade80',
          fontWeight: 'bold',
          fontSize: 14
        }}>
          ✓
        </span>
        <h4 style={{
          fontFamily: 'var(--font-serif)',
          fontSize: 16,
          fontWeight: 400,
          color: '#4ade80',
          margin: 0
        }}>
          The Defense
        </h4>
      </div>
      <p style={{
        fontSize: 15,
        color: 'var(--text)',
        margin: '0 0 16px 0',
        lineHeight: 1.6
      }}>
        {defense}
      </p>
      {demoLinks && demoLinks.length > 0 && (
        <div style={{
          display: 'flex',
          gap: 12,
          flexWrap: 'wrap',
          marginTop: 16
        }}>
          {demoLinks.map((link, idx) => (
            <Link
              key={idx}
              to={link.href}
              style={{
                color: 'var(--gold)',
                textDecoration: 'none',
                fontSize: 13,
                fontWeight: 500,
                fontFamily: 'var(--font-mono)',
                padding: '6px 12px',
                border: '1px solid var(--border)',
                borderRadius: 4,
                display: 'inline-block',
                transition: 'all 0.2s ease'
              }}
            >
              {link.label} →
            </Link>
          ))}
        </div>
      )}
    </div>
  </Card>
);

export default function Falsification() {
  return (
    <div style={{ flex: 1, paddingBottom: 80 }}>
      {/* Hero Section */}
      <section style={{
        background: 'linear-gradient(135deg, var(--navy-dark) 0%, var(--navy-mid) 50%, var(--navy-light) 100%)',
        paddingTop: 80,
        paddingBottom: 80,
        paddingLeft: 24,
        paddingRight: 24,
        marginBottom: 56
      }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--gold)',
            letterSpacing: '0.25em', textTransform: 'uppercase', marginBottom: 20, fontWeight: 500
          }}>
            Three Pillars of Falsification
          </div>
          <h1 style={{
            fontFamily: 'var(--font-serif)',
            fontSize: 64,
            fontWeight: 300,
            color: 'var(--text)',
            margin: '0 0 16px 0',
            letterSpacing: '-1px'
          }}>
            How to Break This
          </h1>
          <p style={{
            fontFamily: 'var(--font-body)',
            fontSize: 20,
            color: 'var(--gold)',
            margin: 0,
            maxWidth: 800,
            lineHeight: 1.7,
            opacity: 0.7
          }}>
            We invite skeptical review. Every claim in Marithmetics includes its own destruction manual.
          </p>
        </div>
      </section>

      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 24px' }}>
      {/* Pillar 1: Base-10 Bias */}
      <div style={{ marginBottom: 56 }}>
        <SectionTitle
          title="Pillar 1: It's Just Numerology / Base-10 Bias"
          subtitle="The rebuttal: base-gauge invariance"
        />
        <AttackDefenseCard
          attack="The results only work because you chose base 10. Swap to binary or hexadecimal and the invariants collapse."
          defense="Run DEMO-64 (base-gauge invariance). The derivation repeats across bases 2, 7, and 16. The same integer closures hold. If true base-invariance fails, the claim is falsified."
          demoLinks={[
            { href: '/demos/demo-64-base-gauge-invariance-integer-selector', label: 'DEMO-64' }
          ]}
        />
      </div>

      {/* Pillar 2: Parameter Tuning */}
      <div style={{ marginBottom: 56 }}>
        <SectionTitle
          title="Pillar 2: You Tuned Parameters to Fit the Data"
          subtitle="The rebuttal: counterfactual ablations"
        />
        <AttackDefenseCard
          attack="The constants (fine structure, electron mass ratios) were cherry-picked after you saw what the physics ought to be. Every '409-class' triple or counterfactual is a free parameter in disguise."
          defense="Counterfactual triples run through the identical pipeline. They explode—fail the same gates that the physical triple passes. Ablation tests are included in the Authority-of-Record bundle. No knobs. No tuning. The method either generalizes or it doesn't."
          demoLinks={[
            { href: '/demos/demo-34-omega-sm-master-flagship', label: 'DEMO-34 (ablations)' },
            { href: '/demos/demo-37-math-sm-master-flagship', label: 'DEMO-37' }
          ]}
        />
      </div>

      {/* Pillar 3: Arbitrary Operators */}
      <div style={{ marginBottom: 56 }}>
        <SectionTitle
          title="Pillar 3: The Operators Are Arbitrary"
          subtitle="The rebuttal: admissibility contracts with illegal controls"
        />
        <AttackDefenseCard
          attack="You could swap in any operator—a sharp cutoff, a signed kernel, some ad-hoc filter—and get similar results. The admissibility claim is just marketing."
          defense="Demos with admissibility contracts include illegal controls. Sharp cutoff, signed Fejér, and other non-admissible kernels are run side-by-side with lawful ones. Illegal operators fail. Only Fejér-class admissible kernels pass the gates. This is falsifiable and tested."
          demoLinks={[
            { href: '/demos/demo-69-oatb-operator-admissibility-transfer-bridge', label: 'DEMO-69 (OATB)' }
          ]}
        />
      </div>

      {/* Evidence Standard */}
      <div style={{ marginBottom: 56 }}>
        <SectionTitle
          title="The Standard of Evidence"
          subtitle="Transparency and reproducibility at every step"
        />
        <Card accent={true}>
          <p style={{
            fontSize: 15,
            color: 'var(--text)',
            margin: '0 0 16px 0',
            lineHeight: 1.6
          }}>
            Every demo in Marithmetics is built on:
          </p>
          <ul style={{
            fontSize: 15,
            color: 'var(--text)',
            margin: '0 0 16px 0',
            lineHeight: 1.8,
            paddingLeft: 20
          }}>
            <li><strong>Stdlib-only code.</strong> No external fitting libraries, no hidden black boxes. Python standard library or minimal dependencies (NumPy). Reproducible in any environment.</li>
            <li><strong>SHA-256 hashing.</strong> Every intermediate result is hashed. The Authority-of-Record bundle is cryptographically sealed. You can rebuild it and verify the hash matches.</li>
            <li><strong>Deterministic execution.</strong> No randomness. Run the same demo twice, get the same output to machine precision. Falsifiability requires reproducibility.</li>
          </ul>
          <MathBlock label="Evidence verification">
            sha256(demo_output) = Authority-of-Record_hash
          </MathBlock>
        </Card>

        <div style={{ marginTop: 24 }}>
          <Link
            to="/methodology"
            style={{
              color: 'var(--gold)',
              textDecoration: 'none',
              fontSize: 14,
              fontWeight: 500,
              fontFamily: 'var(--font-mono)',
              display: 'inline-block',
              padding: '8px 16px',
              border: '1px solid var(--border)',
              borderRadius: 4,
              transition: 'all 0.2s ease'
            }}
          >
            Read the Methodology →
          </Link>
        </div>
      </div>

      {/* Summary Section */}
      <div style={{
        padding: 32,
        background: 'rgba(228,187,124,0.03)',
        border: '1px solid rgba(228,187,124,0.2)',
        borderRadius: 4,
        marginTop: 48
      }}>
        <p style={{
          fontSize: 15,
          color: 'var(--text)',
          margin: 0,
          lineHeight: 1.8
        }}>
          <strong>Falsification is not a weakness—it's the core design principle.</strong> Every assertion in Marithmetics comes with a destruction manual. Run the demos. Check the gates. If you find a base where invariance breaks, a counterfactual that passes, or an illegal operator that works just as well—the claim is false. We mean it.
        </p>
      </div>
      </div>
    </div>
  );
}

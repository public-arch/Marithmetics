import Card from '../components/Card';
import SectionTitle from '../components/SectionTitle';
import MathBlock from '../components/MathBlock';

export default function Methodology() {
  return (
    <div style={{ flex: 1, paddingBottom: 80 }}>
      {/* Hero Section */}
      <section style={{
        background: `linear-gradient(135deg, var(--navy-dark) 0%, var(--navy-mid) 50%, var(--navy-light) 100%)`,
        paddingTop: 80,
        paddingBottom: 80,
        paddingLeft: 24,
        paddingRight: 24,
        marginBottom: 80
      }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--gold)',
            letterSpacing: '0.25em', textTransform: 'uppercase', marginBottom: 20, fontWeight: 500
          }}>
            Audit Protocol · Verification · Reproducibility
          </div>
          <h1 style={{
            fontFamily: 'var(--font-serif)',
            fontSize: 64,
            fontWeight: 300,
            color: 'var(--text)',
            margin: 0,
            marginBottom: 16,
            letterSpacing: '-1px'
          }}>
            Methodology
          </h1>

          <p style={{
            fontFamily: 'var(--font-body)',
            fontSize: 20,
            color: 'var(--gold)',
            margin: 0,
            maxWidth: 700,
            lineHeight: 1.7,
            opacity: 0.7
          }}>
            How to verify every claim. Rebuild everything from source.
          </p>
        </div>
      </section>

      {/* Main Content */}
      <div style={{ maxWidth: 1200, margin: '0 auto', paddingLeft: 24, paddingRight: 24 }}>
        
        {/* Section 1: The Audit Protocol */}
        <section style={{ marginBottom: 80 }}>
          <SectionTitle
            title="The Audit Protocol"
            subtitle="Understanding Authority-of-Record (AoR) and reproducibility"
          />
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: 24,
            marginTop: 24
          }}>
            <Card>
              <h3 style={{
                fontFamily: 'var(--font-serif)',
                fontSize: 18,
                fontWeight: 400,
                color: 'var(--text)',
                margin: '0 0 12px 0'
              }}>
                What is an AoR Bundle?
              </h3>
              <p style={{
                fontSize: 14,
                color: 'var(--text-muted)',
                margin: 0,
                lineHeight: 1.7
              }}>
                Every demo run produces a sealed cryptographic bundle. The bundle contains execution logs, intermediate artifacts, final tables, and a manifest of SHA-256 hashes. Nothing can be modified after sealing.
              </p>
            </Card>

            <Card>
              <h3 style={{
                fontFamily: 'var(--font-serif)',
                fontSize: 18,
                fontWeight: 400,
                color: 'var(--text)',
                margin: '0 0 12px 0'
              }}>
                Deterministic Execution
              </h3>
              <p style={{
                fontSize: 14,
                color: 'var(--text-muted)',
                margin: 0,
                lineHeight: 1.7
              }}>
                Same code + same input = same output + same hash. Every run is independently verifiable. No randomness. No stochastic processes. Reproducible across machines.
              </p>
            </Card>

            <Card>
              <h3 style={{
                fontFamily: 'var(--font-serif)',
                fontSize: 18,
                fontWeight: 400,
                color: 'var(--text)',
                margin: '0 0 12px 0'
              }}>
                Certified Results
              </h3>
              <p style={{
                fontSize: 14,
                color: 'var(--text-muted)',
                margin: 0,
                lineHeight: 1.7
              }}>
                Official papers cite specific AoR bundle hashes. Readers can download the exact bundle, verify the hash locally, and inspect every intermediate step.
              </p>
            </Card>
          </div>
        </section>

        {/* Section 2: Rebuild Everything */}
        <section style={{ marginBottom: 80 }}>
          <SectionTitle
            title="Rebuild Everything"
            subtitle="Three commands to reproduce all results from first principles"
          />
          
          <div style={{ marginTop: 24, marginBottom: 24 }}>
            <p style={{
              fontSize: 16,
              color: 'var(--text-muted)',
              margin: '0 0 24px 0',
              lineHeight: 1.7
            }}>
              The entire pipeline is open-source. Clone the repository and run the master audit suite:
            </p>

            <MathBlock label="Step 1: Run Master Suite">
              {`python -m audits.run_master_suite --verbosity full`}
            </MathBlock>

            <MathBlock label="Step 2: Generate Bundle">
              {`python -m audits.gum_bundle_v30 \\
  --outroot audits/results \\
  --vendor-artifacts \\
  --demos-root demos`}
            </MathBlock>

            <MathBlock label="Step 3: Generate Report">
              {`python gum/gum_report_generator_v32.py \\
  --bundle-dir /path/to/GUM_BUNDLE_v30_*`}
            </MathBlock>

            <p style={{
              fontSize: 14,
              color: 'var(--text-muted)',
              margin: '24px 0 0 0',
              lineHeight: 1.6
            }}>
              Full documentation available in the <code style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 12,
                padding: '2px 6px',
                background: 'rgba(228,187,124,0.1)',
                borderRadius: 3,
                color: 'var(--gold)'
              }}>audits/</code> directory.
            </p>
          </div>
        </section>

        {/* Section 3: What Makes This Different */}
        <section style={{ marginBottom: 80 }}>
          <SectionTitle
            title="What Makes This Different"
            subtitle="Three principles that enable verification"
          />
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: 24,
            marginTop: 24
          }}>
            <Card accent>
              <h3 style={{
                fontFamily: 'var(--font-serif)',
                fontSize: 18,
                fontWeight: 400,
                color: 'var(--gold)',
                margin: '0 0 12px 0'
              }}>
                Deterministic
              </h3>
              <p style={{
                fontSize: 14,
                color: 'var(--text-muted)',
                margin: 0,
                lineHeight: 1.7
              }}>
                Same inputs, same outputs, same hashes. No stochastic elements. No Monte Carlo. No approximation. Every run is identical.
              </p>
            </Card>

            <Card>
              <h3 style={{
                fontFamily: 'var(--font-serif)',
                fontSize: 18,
                fontWeight: 400,
                color: 'var(--text)',
                margin: '0 0 12px 0'
              }}>
                Stdlib-Only
              </h3>
              <p style={{
                fontSize: 14,
                color: 'var(--text-muted)',
                margin: 0,
                lineHeight: 1.7
              }}>
                Core pipeline uses only Python standard library. No hidden dependencies. No proprietary packages. Transparency by design.
              </p>
            </Card>

            <Card>
              <h3 style={{
                fontFamily: 'var(--font-serif)',
                fontSize: 18,
                fontWeight: 400,
                color: 'var(--text)',
                margin: '0 0 12px 0'
              }}>
                Falsifiable by Design
              </h3>
              <p style={{
                fontSize: 14,
                color: 'var(--text-muted)',
                margin: 0,
                lineHeight: 1.7
              }}>
                Every demo includes negative controls. Counterfactual triples are tested under the same pipeline. Legal or illegal. No ambiguity.
              </p>
            </Card>
          </div>
        </section>

        {/* Section 4: Citation Protocol */}
        <section style={{ marginBottom: 48 }}>
          <SectionTitle
            title="Citation Protocol"
            subtitle="How to properly cite Marithmetics results"
          />
          
          <Card style={{ marginTop: 24 }}>
            <p style={{
              fontSize: 14,
              color: 'var(--text-muted)',
              margin: '0 0 16px 0',
              lineHeight: 1.7
            }}>
              A proper citation includes:
            </p>

            <ol style={{
              fontSize: 14,
              color: 'var(--text-muted)',
              margin: 0,
              paddingLeft: 24,
              lineHeight: 1.8
            }}>
              <li><strong style={{ color: 'var(--text)' }}>Demo ID</strong> — e.g., DEMO-64, DEMO-37</li>
              <li><strong style={{ color: 'var(--text)' }}>AoR Bundle Hash</strong> — SHA-256 of the sealed results package</li>
              <li><strong style={{ color: 'var(--text)' }}>Artifact Path</strong> — Location within the bundle (e.g., <code style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 12,
                padding: '2px 4px',
                background: 'rgba(228,187,124,0.1)',
                borderRadius: 2,
                color: 'var(--gold)'
              }}>demos/demo_64/output.txt</code>)</li>
              <li><strong style={{ color: 'var(--text)' }}>File Hash Prefix</strong> — First 8 characters of SHA-256 for tamper detection</li>
            </ol>

            <p style={{
              fontSize: 13,
              color: 'var(--gold)',
              margin: '16px 0 0 0',
              fontFamily: 'var(--font-mono)',
              fontWeight: 500
            }}>
              Example: DEMO-64 | AoR: a7d3f2c1... | Path: demos/demo_64/constants.json | Hash: 3b5e8a9d
            </p>
          </Card>
        </section>
      </div>
    </div>
  );
}

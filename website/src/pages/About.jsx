import { Link } from 'react-router-dom';
import Card from '../components/Card';
import SectionTitle from '../components/SectionTitle';

export default function About() {
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
            Open Research · Open Source · Open Audit
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
            About Marithmetics
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
            A deterministic framework exploring integer-to-physics emergence
          </p>
        </div>
      </section>

      {/* Main Content */}
      <div style={{ maxWidth: 1200, margin: '0 auto', paddingLeft: 24, paddingRight: 24 }}>
        
        {/* Executive Summary */}
        <section style={{ marginBottom: 80 }}>
          <div style={{
            background: 'rgba(228,187,124,0.03)',
            border: '1px solid var(--border-accent)',
            borderRadius: 8,
            padding: 48,
            marginBottom: 0
          }}>
            <p style={{
              fontSize: 18,
              color: 'var(--text)',
              margin: 0,
              lineHeight: 1.8,
              maxWidth: 800
            }}>
              Marithmetics is a deterministic pipeline that derives physical constants and field-theoretic structures from pure number-theoretic constructs. It uses audit-grade methodology—reproducible, cryptographically sealed, and fully open-source—to explore whether integer geometry can explain the Standard Model's empirical constants.
            </p>
          </div>
        </section>

        {/* What This Is */}
        <section style={{ marginBottom: 80 }}>
          <SectionTitle
            title="What This Is"
            subtitle="Core features and scope"
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
                A Computational Pipeline
              </h3>
              <p style={{
                fontSize: 14,
                color: 'var(--text-muted)',
                margin: 0,
                lineHeight: 1.7
              }}>
                Not a theory paper. Every claim is backed by a deterministic, runnable demo. Code is law. Results are sealed in cryptographic Authority-of-Record bundles.
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
                Audit-Grade Methodology
              </h3>
              <p style={{
                fontSize: 14,
                color: 'var(--text-muted)',
                margin: 0,
                lineHeight: 1.7
              }}>
                Every run is independently reproducible. Same inputs, same outputs, same hashes. Core pipeline uses only Python standard library—no hidden dependencies.
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
                Includes negative controls and counterfactual tests. Three explicit attack vectors. Tested under the same deterministic pipeline. Legal or illegal. No ambiguity.
              </p>
            </Card>
          </div>
        </section>

        {/* What This Is Not */}
        <section style={{ marginBottom: 80 }}>
          <SectionTitle
            title="What This Is Not"
            subtitle="Honest disclaimers about scope and status"
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
                Not Peer-Reviewed (Yet)
              </h3>
              <p style={{
                fontSize: 14,
                color: 'var(--text-muted)',
                margin: 0,
                lineHeight: 1.7
              }}>
                This is an open research program. Results are fully auditable and reproducible, but formal peer review is still in progress. The framework is ready for expert scrutiny.
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
                Not Replacing the Standard Model
              </h3>
              <p style={{
                fontSize: 14,
                color: 'var(--text-muted)',
                margin: 0,
                lineHeight: 1.7
              }}>
                Marithmetics produces SM-adjacent predictions from first principles. It's a computational exploration, not a replacement theory. Empirical validation is ongoing.
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
                Not Complete
              </h3>
              <p style={{
                fontSize: 14,
                color: 'var(--text-muted)',
                margin: 0,
                lineHeight: 1.7
              }}>
                This is Tier-A work: open program, actively advancing. Tier-A₁ demos are certified. Higher-order predictions require further validation.
              </p>
            </Card>
          </div>

          {/* Tier System Explanation */}
          <Card accent style={{ marginTop: 24 }}>
            <h4 style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 12,
              fontWeight: 600,
              color: 'var(--gold)',
              margin: '0 0 12px 0',
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              Tier Classification
            </h4>
            
            <div style={{ display: 'grid', gap: 16 }}>
              <div>
                <p style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 13,
                  fontWeight: 600,
                  color: 'var(--text)',
                  margin: '0 0 6px 0'
                }}>
                  Tier-A₁ (Certified)
                </p>
                <p style={{
                  fontSize: 13,
                  color: 'var(--text-muted)',
                  margin: 0
                }}>
                  High-confidence demos with fully audited Authority-of-Record bundles. Published and verified.
                </p>
              </div>
              
              <div>
                <p style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 13,
                  fontWeight: 600,
                  color: 'var(--text)',
                  margin: '0 0 6px 0'
                }}>
                  Tier-A (Open Program)
                </p>
                <p style={{
                  fontSize: 13,
                  color: 'var(--text-muted)',
                  margin: 0
                }}>
                  Active research. Reproducible and auditable, but not yet certified or published.
                </p>
              </div>
              
              <div>
                <p style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 13,
                  fontWeight: 600,
                  color: 'var(--text)',
                  margin: '0 0 6px 0'
                }}>
                  Tier-C (SM-Like Predictions)
                </p>
                <p style={{
                  fontSize: 13,
                  color: 'var(--text-muted)',
                  margin: 0
                }}>
                  Exploratory work. Results are computed deterministically but require higher empirical validation.
                </p>
              </div>
            </div>
          </Card>
        </section>

        {/* About the Author */}
        <section style={{ marginBottom: 80 }}>
          <SectionTitle
            title="About the Author"
            subtitle="The person behind the pipeline"
          />

          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr',
            gap: 32,
            marginTop: 24,
            maxWidth: 800
          }}>
            <Card>
              <p style={{
                fontSize: 16,
                color: 'var(--text)',
                margin: 0,
                lineHeight: 1.9,
                fontFamily: 'var(--font-body)'
              }}>
                Marithmetics was built by <strong style={{ color: 'var(--gold)', fontWeight: 500 }}>Justin Grieshop</strong>, a data scientist and analytics leader based in Ohio. Justin holds a degree from Harvard University and currently serves as Senior Director of AI &amp; Analytics at CBTS, where he leads applied machine learning and business intelligence strategy.
              </p>
            </Card>

            <Card>
              <p style={{
                fontSize: 16,
                color: 'var(--text)',
                margin: 0,
                lineHeight: 1.9,
                fontFamily: 'var(--font-body)'
              }}>
                His professional life spans economics, data science, and AI — but Marithmetics didn't come from any of those disciplines directly. It came from a question he couldn't let go of: <em style={{ color: 'var(--gold)', fontStyle: 'italic' }}>what if the constants of nature aren't arbitrary?</em> The pipeline is the answer to that question, tested the only way he knows how — with code that anyone can run, break, or challenge.
              </p>
            </Card>

            <Card accent>
              <h4 style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 12,
                fontWeight: 600,
                color: 'var(--gold)',
                margin: '0 0 16px 0',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                Philosophy
              </h4>
              <p style={{
                fontSize: 16,
                color: 'var(--text)',
                margin: 0,
                lineHeight: 1.9,
                fontFamily: 'var(--font-body)'
              }}>
                Justin's approach to this work is rooted in the teachings of <strong style={{ color: 'var(--gold)', fontWeight: 500 }}>Thich Nhat Hanh</strong> and the <strong style={{ color: 'var(--gold)', fontWeight: 500 }}>Dhammapada</strong> — a commitment to right effort, intellectual honesty, and the understanding that truth doesn't need to be defended, only presented clearly enough to be tested. The falsification-first posture of Marithmetics isn't just methodological rigor. It reflects a deeper belief: if something is true, it should welcome scrutiny. If it isn't, you should want to know.
              </p>
            </Card>

            <Card>
              <p style={{
                fontSize: 16,
                color: 'var(--text)',
                margin: 0,
                lineHeight: 1.9,
                fontFamily: 'var(--font-body)'
              }}>
                This is open research from one person with a laptop, a question, and the conviction that extraordinary claims require extraordinary transparency. Every demo, every line of code, every hash — it's all here. Not because the work is finished, but because the work is honest.
              </p>
            </Card>
          </div>
        </section>

        {/* Get Involved */}
        <section style={{ marginBottom: 48 }}>
          <SectionTitle
            title="Get Involved"
            subtitle="The research is open. The code is open. Help us falsify this."
          />
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
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
                Clone the Repository
              </h3>
              <p style={{
                fontSize: 14,
                color: 'var(--text-muted)',
                margin: '0 0 16px 0',
                lineHeight: 1.6
              }}>
                Full source code is available. Run the demos. Verify the results. Check our math.
              </p>
              <a
                href="https://github.com/public-arch/Marithmetics"
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  color: 'var(--gold)',
                  textDecoration: 'none',
                  fontSize: 13,
                  fontWeight: 500,
                  fontFamily: 'var(--font-mono)',
                  display: 'inline-block',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.target.style.transform = 'translateX(4px)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.transform = 'translateX(0)';
                }}
              >
                View on GitHub →
              </a>
            </Card>

            <Card>
              <h3 style={{
                fontFamily: 'var(--font-serif)',
                fontSize: 18,
                fontWeight: 400,
                color: 'var(--text)',
                margin: '0 0 12px 0'
              }}>
                Run the Demos
              </h3>
              <p style={{
                fontSize: 14,
                color: 'var(--text-muted)',
                margin: '0 0 16px 0',
                lineHeight: 1.6
              }}>
                Every claim has a deterministic demo. Standard library Python. Reproducible in minutes.
              </p>
              <Link
                to="/demos"
                style={{
                  color: 'var(--gold)',
                  textDecoration: 'none',
                  fontSize: 13,
                  fontWeight: 500,
                  fontFamily: 'var(--font-mono)',
                  display: 'inline-block',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.target.style.transform = 'translateX(4px)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.transform = 'translateX(0)';
                }}
              >
                Explore 29 demos →
              </Link>
            </Card>

            <Card>
              <h3 style={{
                fontFamily: 'var(--font-serif)',
                fontSize: 18,
                fontWeight: 400,
                color: 'var(--text)',
                margin: '0 0 12px 0'
              }}>
                Falsify It
              </h3>
              <p style={{
                fontSize: 14,
                color: 'var(--text-muted)',
                margin: '0 0 16px 0',
                lineHeight: 1.6
              }}>
                Find a flaw. Challenge an assumption. Propose a test. That's how science works.
              </p>
              <Link
                to="/falsification"
                style={{
                  color: 'var(--gold)',
                  textDecoration: 'none',
                  fontSize: 13,
                  fontWeight: 500,
                  fontFamily: 'var(--font-mono)',
                  display: 'inline-block',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.target.style.transform = 'translateX(4px)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.transform = 'translateX(0)';
                }}
              >
                Falsification framework →
              </Link>
            </Card>
          </div>
        </section>

        {/* Research Status */}
        <section style={{
          background: 'rgba(228,187,124,0.03)',
          border: '1px solid var(--border-accent)',
          borderRadius: 8,
          padding: 48,
          textAlign: 'center'
        }}>
          <h2 style={{
            fontFamily: 'var(--font-serif)',
            fontSize: 28,
            fontWeight: 400,
            color: 'var(--text)',
            margin: '0 0 16px 0'
          }}>
            Status: Active Research
          </h2>

          <p style={{
            fontSize: 16,
            color: 'var(--text-muted)',
            margin: 0,
            maxWidth: 700,
            marginLeft: 'auto',
            marginRight: 'auto',
            lineHeight: 1.7
          }}>
            Started 2024. Tier-A₁ demos published. Peer review in progress. All code, data, and Authority-of-Record bundles public and auditable.
          </p>

          <div style={{
            marginTop: 32,
            paddingTop: 32,
            borderTop: '1px solid rgba(228,187,124,0.15)'
          }}>
            <p style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 12,
              color: 'var(--text-muted)',
              margin: '0 0 8px 0',
              letterSpacing: '0.1em',
              textTransform: 'uppercase'
            }}>
              Contact
            </p>
            <a
              href="mailto:Public@Marithmetics.com"
              style={{
                fontFamily: 'var(--font-serif)',
                fontSize: 18,
                color: 'var(--gold)',
                textDecoration: 'none',
                fontWeight: 400,
                transition: 'opacity 0.2s ease'
              }}
              onMouseEnter={(e) => { e.target.style.opacity = '0.7'; }}
              onMouseLeave={(e) => { e.target.style.opacity = '1'; }}
            >
              Public@Marithmetics.com
            </a>
          </div>
        </section>
      </div>
    </div>
  );
}

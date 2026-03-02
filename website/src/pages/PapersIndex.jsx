import Card from '../components/Card';
import Badge from '../components/Badge';
import SectionTitle from '../components/SectionTitle';
import papers from '../data/papers.json';

const DISCIPLINE_LABELS = {
  governance: 'Governance',
  number_theory: 'Number Theory',
  physics: 'Physics',
  authority: 'Authority Record'
};

const DISCIPLINE_ORDER = ['governance', 'number_theory', 'physics', 'authority'];

const PaperCard = ({ paper }) => (
  <Card style={{
    display: 'flex',
    flexDirection: 'column',
    height: '100%'
  }}>
    <div style={{ marginBottom: 12 }}>
      <Badge 
        category={paper.discipline} 
        text={DISCIPLINE_LABELS[paper.discipline]} 
      />
    </div>
    <h3 style={{
      fontFamily: 'var(--font-serif)',
      fontSize: 16,
      fontWeight: 400,
      color: 'var(--text)',
      margin: '0 0 12px 0',
      lineHeight: 1.5
    }}>
      {paper.title}
    </h3>
    <p style={{
      fontSize: 12,
      color: 'var(--text-muted)',
      margin: '0 0 16px auto',
      fontFamily: 'var(--font-mono)',
      marginTop: 'auto'
    }}>
      {paper.filename}
    </p>
    <a
      href={`https://github.com/public-arch/Marithmetics/raw/main/publication_spine/${paper.path}`}
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
      Download PDF →
    </a>
  </Card>
);

export default function PapersIndex() {
  const groupedPapers = DISCIPLINE_ORDER.reduce((acc, discipline) => {
    const filtered = papers.filter(p => p.discipline === discipline);
    if (filtered.length > 0) {
      acc[discipline] = filtered;
    }
    return acc;
  }, {});

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
            Governance · Number Theory · Physics · Authority
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
            Publication Spine
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
            Authority documents organized by discipline. Each paper cites specific AoR bundle hashes.
          </p>
        </div>
      </section>

      {/* Main Content */}
      <div style={{ maxWidth: 1200, margin: '0 auto', paddingLeft: 24, paddingRight: 24 }}>
        
        {/* Papers by Discipline */}
        {Object.entries(groupedPapers).map(([discipline, disciplinePapers]) => (
          <section key={discipline} style={{ marginBottom: 80 }}>
            <SectionTitle
              title={DISCIPLINE_LABELS[discipline]}
              subtitle={`${disciplinePapers.length} paper${disciplinePapers.length !== 1 ? 's' : ''}`}
            />
            
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
              gap: 24,
              marginTop: 24
            }}>
              {disciplinePapers.map(paper => (
                <PaperCard key={paper.id} paper={paper} />
              ))}
            </div>
          </section>
        ))}

        {/* Info Section */}
        <section style={{
          background: 'rgba(228,187,124,0.03)',
          border: '1px solid var(--border-accent)',
          borderRadius: 8,
          padding: 48,
          marginBottom: 48,
          textAlign: 'center'
        }}>
          <h2 style={{
            fontFamily: 'var(--font-serif)',
            fontSize: 28,
            fontWeight: 400,
            color: 'var(--text)',
            margin: '0 0 16px 0'
          }}>
            All PDFs Available in GitHub
          </h2>
          
          <p style={{
            fontSize: 16,
            color: 'var(--text-muted)',
            margin: '0 0 24px 0',
            maxWidth: 700,
            marginLeft: 'auto',
            marginRight: 'auto',
            lineHeight: 1.7
          }}>
            PDFs are available in the official GitHub repository under <code style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 13,
              padding: '2px 6px',
              background: 'rgba(228,187,124,0.1)',
              borderRadius: 3,
              color: 'var(--gold)'
            }}>publication_spine/</code>
          </p>
          
          <a
            href="https://github.com/public-arch/Marithmetics/tree/main/publication_spine"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: 'var(--gold)',
              textDecoration: 'none',
              fontSize: 14,
              fontWeight: 500,
              fontFamily: 'var(--font-mono)',
              display: 'inline-block',
              marginTop: 8,
              transition: 'all 0.3s ease'
            }}
            onMouseEnter={(e) => {
              e.target.style.transform = 'translateX(4px)';
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = 'translateX(0)';
            }}
          >
            View repository →
          </a>
        </section>
      </div>
    </div>
  );
}

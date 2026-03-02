import { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import Card from '../components/Card';
import Badge from '../components/Badge';
import SectionTitle from '../components/SectionTitle';
import demos from '../data/demos.json';
import { getCategoryLabel } from '../utils/categories';

// Extract unique categories
const CATEGORIES = ['All', ...new Set(demos.map(d => d.category))].sort((a, b) => {
  if (a === 'All') return -1;
  if (b === 'All') return 1;
  return a.localeCompare(b);
});

export default function DemoIndex() {
  const [selectedCategory, setSelectedCategory] = useState('All');

  const filteredDemos = useMemo(() => {
    if (selectedCategory === 'All') {
      return demos;
    }
    return demos.filter(d => d.category === selectedCategory);
  }, [selectedCategory]);

  const demoCount = demos.length;
  const categoryCount = new Set(demos.map(d => d.category)).size;

  return (
    <div style={{ flex: 1, paddingBottom: 80 }}>
      {/* Hero Section */}
      <section style={{
        background: 'linear-gradient(135deg, var(--navy-dark) 0%, var(--navy-mid) 50%, var(--navy-light) 100%)',
        paddingTop: 80,
        paddingBottom: 80,
        paddingLeft: 24,
        paddingRight: 24,
        marginBottom: 48
      }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--gold)',
            letterSpacing: '0.25em', textTransform: 'uppercase', marginBottom: 20, fontWeight: 500
          }}>
            Reproducible · Deterministic · Falsifiable
          </div>
          <h1 style={{
            fontFamily: 'var(--font-serif)',
            fontSize: 64,
            fontWeight: 300,
            color: 'var(--text)',
            margin: '0 0 16px 0',
            letterSpacing: '-1px'
          }}>
            Demo Suite
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
            {demoCount} deterministic demos across {categoryCount} domains. Every result sealed in a cryptographic Authority-of-Record.
          </p>
        </div>
      </section>

      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 24px' }}>
      {/* Category Filter */}
      <div style={{ marginBottom: 40 }}>
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 12,
          alignItems: 'center'
        }}>
          {CATEGORIES.map(category => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              style={{
                padding: '8px 16px',
                borderRadius: 4,
                border: `1px solid ${selectedCategory === category ? 'var(--gold)' : 'var(--border)'}`,
                background: selectedCategory === category ? 'rgba(228,187,124,0.1)' : 'transparent',
                color: selectedCategory === category ? 'var(--gold)' : 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
                fontSize: 13,
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              {category === 'All' ? 'All' : getCategoryLabel(category)}
            </button>
          ))}
        </div>
      </div>

      {/* Results Count */}
      <div style={{ marginBottom: 24 }}>
        <p style={{
          fontSize: 14,
          color: 'var(--text-muted)',
          margin: 0,
          fontFamily: 'var(--font-mono)'
        }}>
          {filteredDemos.length} demo{filteredDemos.length !== 1 ? 's' : ''} {selectedCategory !== 'All' && `in ${getCategoryLabel(selectedCategory)}`}
        </p>
      </div>

      {/* Demo Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
        gap: 24,
        marginTop: 32
      }}>
        {filteredDemos.map(demo => (
          <Link
            key={demo.id}
            to={`/demos/${demo.slug}`}
            style={{ textDecoration: 'none', color: 'inherit' }}
          >
            <Card style={{
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              height: '100%',
              display: 'flex',
              flexDirection: 'column'
            }}>
              {/* Demo ID and Category */}
              <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
                <Badge text={demo.id} ok={true} />
                <Badge category={demo.category} />
              </div>

              {/* Title */}
              <h3 style={{
                fontFamily: 'var(--font-serif)',
                fontSize: 17,
                fontWeight: 400,
                color: 'var(--text)',
                margin: '0 0 12px 0',
                lineHeight: 1.3
              }}>
                {demo.shortTitle}
              </h3>

              {/* Description */}
              <p style={{
                fontSize: 13,
                color: 'var(--text-muted)',
                margin: '0 0 16px 0',
                lineHeight: 1.6,
                flex: 1
              }}>
                {demo.description}
              </p>

              {/* Status and Flagship */}
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 'auto' }}>
                <Badge text={demo.status} ok={demo.status === 'certified'} />
                {demo.flagship && (
                  <Badge text="Flagship" ok={true} />
                )}
              </div>
            </Card>
          </Link>
        ))}
      </div>

      {filteredDemos.length === 0 && (
        <div style={{
          textAlign: 'center',
          padding: '60px 24px',
          color: 'var(--text-muted)'
        }}>
          <p style={{ fontSize: 16, margin: 0 }}>No demos found in this category.</p>
        </div>
      )}
      </div>
    </div>
  );
}

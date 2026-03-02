import { CATEGORY_COLORS, getCategoryLabel } from '../utils/categories';

export default function Badge({ ok, text, category }) {
  if (category) {
    return (
      <span style={{
        display: 'inline-flex', alignItems: 'center', gap: 6,
        padding: '4px 12px', borderRadius: 4,
        background: `${CATEGORY_COLORS[category] || '#e4bb7c'}20`,
        color: CATEGORY_COLORS[category] || '#e4bb7c',
        fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 500
      }}>{text || getCategoryLabel(category)}</span>
    );
  }
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '4px 12px', borderRadius: 4,
      background: ok ? 'rgba(74,222,128,0.12)' : 'rgba(248,113,113,0.12)',
      color: ok ? '#4ade80' : '#f87171',
      fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 500
    }}>{ok ? '✓' : '✗'} {text || (ok ? 'PASS' : 'FAIL')}</span>
  );
}

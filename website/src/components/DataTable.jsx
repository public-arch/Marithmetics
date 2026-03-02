export default function DataTable({ headers, rows, highlightFn }) {
  return (
    <div style={{ overflow: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: 13 }}>
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th key={i} style={{
                textAlign: 'left', padding: '12px 16px',
                borderBottom: '1px solid var(--border-accent)',
                color: 'var(--gold)', fontWeight: 500, fontSize: 11,
                textTransform: 'uppercase', letterSpacing: '0.1em'
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} style={{
              background: highlightFn && highlightFn(row, i) ? 'var(--card-accent-bg)' : 'transparent'
            }}>
              {row.map((cell, j) => (
                <td key={j} style={{
                  padding: '10px 16px',
                  borderBottom: '1px solid rgba(255,255,255,0.05)',
                  color: 'rgba(255,255,255,0.8)'
                }}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

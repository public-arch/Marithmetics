import { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';

/* ═══════════════════════════════════════════
   Core computation — exact port from original
   ═══════════════════════════════════════════ */

function digitalRoot(num, base) {
  const modulus = base - 1;
  if (num === 0) return 0;
  const result = num % modulus;
  return result === 0 ? modulus : result;
}

function buildGrid(base) {
  const maxBase = base - 1;
  const rows = [];
  for (let b = 1; b <= maxBase; b++) {
    const row = [b];
    for (let power = 1; power <= 12; power++) {
      const value = Math.pow(b, power);
      row.push(digitalRoot(value, base));
    }
    rows.push(row);
  }
  return rows;
}

/* ── Pattern detection — all 17, exact from original ── */

function findMultiplicativeInverses(grid, base) {
  const mod = base - 1;
  const pairs = [];
  for (let a = 2; a < mod; a++) {
    for (let b = a + 1; b <= mod; b++) {
      if ((a * b) % mod === 1) pairs.push([a, b]);
    }
  }
  return pairs;
}

function findAttractors(grid, base) {
  const mod = base - 1;
  const attractors = [];
  for (let b = 2; b <= mod; b++) {
    const row = grid[b - 1];
    if (!row) continue;
    const last = row[row.length - 1];
    const second = row[row.length - 2];
    const third = row[row.length - 3];
    if (last === second && last === third) attractors.push(b);
  }
  return attractors;
}

function findCycleLength(grid, baseIndex) {
  const row = grid[baseIndex];
  if (!row) return 1;
  const last = row[row.length - 1];
  let isAttractor = true;
  for (let i = Math.max(2, row.length - 6); i < row.length; i++) {
    if (row[i] !== last) { isAttractor = false; break; }
  }
  if (isAttractor) return 1;
  for (let len = 1; len <= 12; len++) {
    let isCycle = true;
    const checkLen = Math.min(6, row.length - len - 1);
    for (let i = 1; i <= checkLen; i++) {
      if (row[i] !== row[i + len]) { isCycle = false; break; }
    }
    if (isCycle && checkLen > 0) return len;
  }
  return 12;
}

function findFrequentValues(grid, base) {
  const valueCounts = {};
  const valuePositions = {};
  for (let r = 0; r < grid.length; r++) {
    for (let c = 1; c < grid[r].length; c++) {
      const val = grid[r][c];
      valueCounts[val] = (valueCounts[val] || 0) + 1;
      if (!valuePositions[val]) valuePositions[val] = [];
      valuePositions[val].push({ base: r + 1, power: c });
    }
  }
  const mod = base - 1;
  const totalCells = grid.length * 12;
  return Object.entries(valueCounts)
    .filter(([val, count]) => {
      if (val === '1' && count / totalCells > 0.4) return false;
      if (parseInt(val) === mod && count / totalCells > 0.4) return false;
      return count >= 3;
    })
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([val, count]) => ({ value: parseInt(val), count, positions: valuePositions[val] }));
}

function findAllDiagonals(grid) {
  const diagonals = [];
  // Down-right
  for (let startRow = 0; startRow < grid.length; startRow++) {
    for (let startCol = 1; startCol < 13; startCol++) {
      const diagonal = [];
      let r = startRow, c = startCol;
      while (r < grid.length && c < 13) {
        diagonal.push({ base: r + 1, power: c, value: grid[r][c] });
        r++; c++;
      }
      if (diagonal.length >= 3) {
        const vc = {};
        diagonal.forEach(d => { vc[d.value] = (vc[d.value] || 0) + 1; });
        for (const [val, count] of Object.entries(vc)) {
          if (count >= 3) {
            diagonals.push({
              direction: 'down-right', value: parseInt(val),
              cells: diagonal.filter(d => d.value === parseInt(val)), count
            });
          }
        }
      }
    }
  }
  // Up-right
  for (let startRow = 0; startRow < grid.length; startRow++) {
    for (let startCol = 1; startCol < 13; startCol++) {
      const diagonal = [];
      let r = startRow, c = startCol;
      while (r >= 0 && c < 13) {
        diagonal.push({ base: r + 1, power: c, value: grid[r][c] });
        r--; c++;
      }
      if (diagonal.length >= 3) {
        const vc = {};
        diagonal.forEach(d => { vc[d.value] = (vc[d.value] || 0) + 1; });
        for (const [val, count] of Object.entries(vc)) {
          if (count >= 3) {
            diagonals.push({
              direction: 'up-right', value: parseInt(val),
              cells: diagonal.filter(d => d.value === parseInt(val)), count
            });
          }
        }
      }
    }
  }
  // Deduplicate
  const unique = [];
  const seen = new Set();
  for (const diag of diagonals) {
    const key = diag.cells.map(c => `${c.base}-${c.power}`).sort().join('|');
    if (!seen.has(key)) { seen.add(key); unique.push(diag); }
  }
  return unique.sort((a, b) => b.count - a.count).slice(0, 10);
}

function findOscillations(grid) {
  const patterns = [];
  for (let row = 0; row < grid.length; row++) {
    const rowData = grid[row].slice(1);
    for (let i = 0; i < rowData.length - 3; i++) {
      if (rowData[i] === rowData[i + 2] && rowData[i + 1] === rowData[i + 3] && rowData[i] !== rowData[i + 1]) {
        patterns.push({
          type: 'row', base: row + 1, startPower: i + 1,
          values: [rowData[i], rowData[i + 1]],
          cells: [
            { base: row + 1, power: i + 1 }, { base: row + 1, power: i + 2 },
            { base: row + 1, power: i + 3 }, { base: row + 1, power: i + 4 }
          ]
        });
      }
    }
  }
  for (let col = 1; col < 13; col++) {
    const colData = grid.map(r => r[col]);
    for (let i = 0; i < colData.length - 3; i++) {
      if (colData[i] === colData[i + 2] && colData[i + 1] === colData[i + 3] && colData[i] !== colData[i + 1]) {
        patterns.push({
          type: 'column', base: i + 1, startPower: col,
          values: [colData[i], colData[i + 1]],
          cells: [
            { base: i + 1, power: col }, { base: i + 2, power: col },
            { base: i + 3, power: col }, { base: i + 4, power: col }
          ]
        });
      }
    }
  }
  return patterns;
}

function findDivergentMirrors(grid, base) {
  const mirrors = [];
  for (let col = 1; col <= 12; col++) {
    const colData = grid.map(r => r[col]);
    for (let center = 1; center < colData.length - 2; center++) {
      const maxR = Math.min(center, colData.length - center - 1);
      let matchLen = 0;
      for (let r = 1; r <= maxR; r++) {
        if (colData[center - r] === colData[center + r]) matchLen = r;
        else break;
      }
      if (matchLen >= 2) {
        mirrors.push({
          power: col, type: 'single-axis', centerBase: center + 1,
          radius: matchLen, data: colData
        });
      }
      if (center < colData.length - 1 && colData[center] === colData[center + 1]) {
        const maxR2 = Math.min(center, colData.length - center - 2);
        let matchLen2 = 0;
        for (let r = 1; r <= maxR2; r++) {
          if (colData[center - r] === colData[center + 1 + r]) matchLen2 = r;
          else break;
        }
        if (matchLen2 >= 2) {
          mirrors.push({
            power: col, type: 'double-axis', centerBase1: center + 1,
            centerBase2: center + 2, radius: matchLen2, data: colData
          });
        }
      }
    }
  }
  const best = [];
  const usedCols = new Set();
  mirrors.sort((a, b) => b.radius - a.radius);
  for (const m of mirrors) {
    if (!usedCols.has(m.power)) { best.push(m); usedCols.add(m.power); }
  }
  return best;
}

function findRepeatingColumnSegments(grid) {
  const repeating = [];
  for (let col = 1; col <= 12; col++) {
    const colData = grid.map(r => r[col]);
    for (let segLen = 2; segLen <= Math.floor(colData.length / 2); segLen++) {
      const segment = colData.slice(0, segLen);
      let repeatCount = 0, allMatch = true;
      for (let i = 0; i < colData.length; i += segLen) {
        const cur = colData.slice(i, i + segLen);
        if (cur.length !== segLen) break;
        if (cur.every((v, j) => v === segment[j])) repeatCount++;
        else { allMatch = false; break; }
      }
      if (allMatch && repeatCount >= 2) {
        repeating.push({ power: col, segmentLength: segLen, segment, repeatCount, data: colData });
        break;
      }
    }
  }
  return repeating;
}

function findIdenticalColumns(grid) {
  const twins = [];
  const checked = new Set();
  for (let c1 = 1; c1 <= 12; c1++) {
    if (checked.has(c1)) continue;
    const d1 = grid.map(r => r[c1]);
    const group = [c1];
    for (let c2 = c1 + 1; c2 <= 12; c2++) {
      if (checked.has(c2)) continue;
      const d2 = grid.map(r => r[c2]);
      if (d1.every((v, i) => v === d2[i])) { group.push(c2); checked.add(c2); }
    }
    if (group.length > 1) { checked.add(c1); twins.push({ powers: group, data: d1 }); }
  }
  return twins;
}

function findColumnAttractors(grid) {
  const attractors = [];
  for (let col = 1; col <= 12; col++) {
    const colData = grid.map(r => r[col]);
    const vc = {};
    colData.forEach(v => { vc[v] = (vc[v] || 0) + 1; });
    for (const [value, count] of Object.entries(vc)) {
      if (count / colData.length > 0.5) {
        attractors.push({ power: col, value: parseInt(value), count, percentage: (count / colData.length * 100).toFixed(1) });
        break;
      }
    }
  }
  return attractors;
}

function findComplementaryColumns(grid, base) {
  const mod = base - 1;
  const complements = [];
  const checked = new Set();
  for (let c1 = 1; c1 <= 12; c1++) {
    if (checked.has(c1)) continue;
    const d1 = grid.map(r => r[c1]);
    for (let c2 = c1 + 1; c2 <= 12; c2++) {
      if (checked.has(c2)) continue;
      const d2 = grid.map(r => r[c2]);
      const minLen = Math.min(d1.length, d2.length);
      const threshold = minLen * 0.6;
      // Sum to modulus
      let countSumMod = 0;
      for (let i = 0; i < minLen; i++) { if (d1[i] + d2[i] === mod) countSumMod++; }
      if (countSumMod >= threshold) {
        complements.push({ power1: c1, power2: c2, type: 'sum-to-modulus', strength: (countSumMod / minLen * 100).toFixed(0) + '%' });
        checked.add(c1); checked.add(c2); break;
      }
      // Multiplicative inverse
      let countMultInv = 0;
      for (let i = 0; i < minLen; i++) {
        if (d1[i] !== 0 && d2[i] !== 0 && (d1[i] * d2[i]) % mod === 1) countMultInv++;
      }
      if (countMultInv >= threshold) {
        complements.push({ power1: c1, power2: c2, type: 'multiplicative-inverse', strength: (countMultInv / minLen * 100).toFixed(0) + '%' });
        checked.add(c1); checked.add(c2); break;
      }
      // Constant difference
      let constDiff = true, diff = d2[0] - d1[0];
      for (let i = 1; i < minLen; i++) { if (d2[i] - d1[i] !== diff) { constDiff = false; break; } }
      if (constDiff && diff !== 0) {
        complements.push({ power1: c1, power2: c2, type: 'constant-difference', difference: diff, strength: '100%' });
        checked.add(c1); checked.add(c2); break;
      }
    }
  }
  return complements;
}

function findArithmeticColumnProgressions(grid) {
  const progressions = [];
  for (let col = 1; col <= 12; col++) {
    const colData = grid.map(r => r[col]);
    for (let start = 0; start < colData.length - 2; start++) {
      const diff = colData[start + 1] - colData[start];
      let length = 2;
      for (let i = start + 2; i < colData.length; i++) {
        if (colData[i] - colData[i - 1] === diff) length++;
        else break;
      }
      if (length >= 3) {
        progressions.push({ power: col, start, length, difference: diff, sequence: colData.slice(start, start + length) });
        start += length - 1;
      }
    }
  }
  return progressions;
}

function getCollapsedColumnRoots(grid, base) {
  const roots = [];
  for (let col = 1; col <= 12; col++) {
    let sum = 0;
    grid.forEach(r => { sum += r[col]; });
    roots.push({ power: col, sum, digitalRoot: digitalRoot(sum, base) });
  }
  return roots;
}

function getCollapsedRowRoots(grid, base) {
  return grid.map((row, i) => {
    let sum = 0;
    for (let c = 1; c <= 12; c++) sum += row[c];
    return { base: i + 1, sum, digitalRoot: digitalRoot(sum, base) };
  });
}

function detectSequencePattern(values) {
  if (!values.length) return 'Empty';
  if (values.every(v => v === values[0])) return `Constant: all ${values[0]}`;
  for (let len = 1; len <= Math.floor(values.length / 2); len++) {
    const pat = values.slice(0, len);
    if (values.every((v, i) => v === pat[i % len]))
      return `Repeating [${pat.join(', ')}] with period ${len}`;
  }
  const d = values[1] - values[0];
  if (values.every((v, i) => i === 0 || v - values[i - 1] === d))
    return `Arithmetic progression (d=${d > 0 ? '+' : ''}${d})`;
  return `${[...new Set(values)].length} unique values`;
}

/* ═══════════════════════════════════════════
   Color system — restrained gold palette
   ═══════════════════════════════════════════ */

const GOLD = '#e4bb7c';
const GOLD_DIM = 'rgba(228,187,124,0.15)';
const GOLD_MID = 'rgba(228,187,124,0.28)';
const GOLD_STRONG = 'rgba(228,187,124,0.50)';
const SILVER = 'rgba(140,170,210,0.30)';
const SILVER_STRONG = 'rgba(140,170,210,0.50)';
const WARM = 'rgba(210,150,90,0.40)';
const COOL = 'rgba(90,150,210,0.35)';
const DEEP = 'rgba(160,120,200,0.35)';
const AXIS = 'rgba(228,100,100,0.45)';

/* ═══════════════════════════════════════════
   All 17 patterns
   ═══════════════════════════════════════════ */

const PATTERNS = [
  { id: 'inverse-1', label: 'Primary Inverse Pair', group: 'Horizontal',
    desc: 'The first multiplicative inverse pair: a × b ≡ 1 (mod base−1). Their power sequences are algebraically coupled.' },
  { id: 'inverse-2', label: 'Secondary Inverse Pair', group: 'Horizontal',
    desc: 'The second multiplicative inverse pair, if it exists. All pairs in the base are listed.' },
  { id: 'attractors', label: 'Attractor Bases', group: 'Horizontal',
    desc: 'Bases whose powers converge to a fixed digital root, regardless of exponent.' },
  { id: 'cycles', label: 'Cycle Lengths', group: 'Horizontal',
    desc: 'The minimal repeating period of each base\'s digital root power sequence. Highlighted cells show one full cycle.' },
  { id: 'identity', label: 'Identity Element', group: 'Horizontal',
    desc: 'Positions where base^power ≡ 1 — the multiplicative identity in mod arithmetic. These mark cycle completion.' },
  { id: 'echoes', label: 'Frequent Values', group: 'Horizontal',
    desc: 'The most common values in the table — structural echoes of the modular group.' },
  { id: 'diagonals', label: 'Diagonal Chains', group: 'Horizontal',
    desc: 'Values forming diagonal lines across the table — cross-base, cross-power invariants.' },
  { id: 'oscillations', label: 'Oscillations', group: 'Horizontal',
    desc: 'A-B-A-B alternating patterns in rows and columns — order-2 cyclic behaviour.' },
  { id: 'column-mirrors', label: 'Divergent Mirrors', group: 'Vertical',
    desc: 'Columns where values mirror symmetrically from a central axis — palindromic vertical structure.' },
  { id: 'column-repeats', label: 'Column Tiling', group: 'Vertical',
    desc: 'Columns where a fixed segment tiles vertically, repeating exactly. This tiling extends to infinity.' },
  { id: 'column-twins', label: 'Identical Columns', group: 'Vertical',
    desc: 'Distinct powers producing identical digital root sequences — structural redundancy.' },
  { id: 'column-attractors', label: 'Column Attractors', group: 'Vertical',
    desc: 'Columns dominated by a single value (>50%). The value acts as a vertical attractor.' },
  { id: 'column-complements', label: 'Column Complements', group: 'Vertical',
    desc: 'Paired columns with mathematical relationships: sum-to-modulus, multiplicative inverse, or constant difference.' },
  { id: 'column-progressions', label: 'Column Progressions', group: 'Vertical',
    desc: 'Consecutive values in a column forming arithmetic sequences — linear structure in the vertical.' },
  { id: 'collapsed-columns', label: 'Collapsed Columns', group: 'Collapsed',
    desc: 'Sum each column, take its digital root — the table\'s vertical signature.' },
  { id: 'collapsed-rows', label: 'Collapsed Rows', group: 'Collapsed',
    desc: 'Sum each row, take its digital root — the table\'s horizontal signature.' },
];

const BASE_OPTIONS = [
  { value: 10, label: 'Base 10' },
  { value: 3, label: 'Base 3' },
  { value: 4, label: 'Base 4' },
  { value: 5, label: 'Base 5' },
  { value: 6, label: 'Base 6' },
  { value: 7, label: 'Base 7' },
  { value: 8, label: 'Base 8' },
  { value: 9, label: 'Base 9' },
  { value: 11, label: 'Base 11' },
  { value: 12, label: 'Base 12' },
  { value: 13, label: 'Base 13' },
  { value: 14, label: 'Base 14' },
  { value: 15, label: 'Base 15' },
  { value: 16, label: 'Base 16' },
];

/* ═══════════════════════════════════════════
   Table component
   ═══════════════════════════════════════════ */

function DRPTTable({ grid, highlightFn }) {
  return (
    <div style={{ overflowX: 'auto', margin: '24px 0' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: 13 }}>
        <thead>
          <tr>
            <th style={thStyle}>Base</th>
            {Array.from({ length: 12 }, (_, i) => <th key={i} style={thStyle}>^{i + 1}</th>)}
          </tr>
        </thead>
        <tbody>
          {grid.map((row, ri) => (
            <tr key={ri}>
              <td style={{ ...tdStyle, fontWeight: 700, background: 'rgba(228,187,124,0.06)', color: GOLD }}>{row[0]}</td>
              {row.slice(1).map((val, ci) => {
                const bg = highlightFn ? highlightFn(ri + 1, ci + 1, val) : null;
                return (
                  <td key={ci} style={{
                    ...tdStyle,
                    background: bg || 'transparent',
                    fontWeight: bg ? 700 : 400,
                    color: bg ? '#fff' : 'var(--text-muted)',
                  }}>{val}</td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const thStyle = {
  padding: '8px 6px', textAlign: 'center',
  borderBottom: '2px solid rgba(228,187,124,0.25)',
  color: GOLD, fontWeight: 700, fontSize: 11, letterSpacing: '0.05em'
};

const tdStyle = {
  padding: '5px 4px', textAlign: 'center',
  border: '1px solid rgba(228,187,124,0.08)', fontSize: 13
};

/* ═══════════════════════════════════════════
   Highlight functions — one per pattern
   ═══════════════════════════════════════════ */

function getHighlightFn(pattern, grid, base) {
  const mod = base - 1;

  switch (pattern) {
    case 'inverse-1': {
      const pairs = findMultiplicativeInverses(grid, base);
      if (!pairs.length) return null;
      const [a, b] = pairs[0];
      return (base_, p, v) => {
        if (base_ === a) return GOLD_STRONG;
        if (base_ === b) return COOL;
        return null;
      };
    }
    case 'inverse-2': {
      const pairs = findMultiplicativeInverses(grid, base);
      const pair = pairs.length > 1 ? pairs[1] : pairs[0];
      if (!pair) return null;
      return (base_, p, v) => {
        if (base_ === pair[0]) return GOLD_STRONG;
        if (base_ === pair[1]) return COOL;
        return null;
      };
    }
    case 'attractors': {
      const att = new Set(findAttractors(grid, base));
      return (b) => att.has(b) ? SILVER : null;
    }
    case 'cycles': {
      return (b, p) => {
        const len = findCycleLength(grid, b - 1);
        return p <= len ? GOLD_DIM : null;
      };
    }
    case 'identity': {
      return (b, p, v) => v === 1 ? GOLD_STRONG : null;
    }
    case 'echoes': {
      const freq = findFrequentValues(grid, base);
      const colors = [GOLD_STRONG, DEEP, SILVER_STRONG];
      return (b, p, v) => {
        for (let i = 0; i < Math.min(3, freq.length); i++) {
          if (v === freq[i].value) return colors[i];
        }
        return null;
      };
    }
    case 'diagonals': {
      const diags = findAllDiagonals(grid).slice(0, 2);
      const colors = [GOLD_MID, COOL];
      const cellSets = diags.map(d => new Set(d.cells.map(c => `${c.base}-${c.power}`)));
      return (b, p) => {
        for (let i = 0; i < cellSets.length; i++) {
          if (cellSets[i].has(`${b}-${p}`)) return colors[i];
        }
        return null;
      };
    }
    case 'oscillations': {
      const osc = findOscillations(grid);
      const cells = new Set();
      osc.forEach(o => o.cells.forEach(c => cells.add(`${c.base}-${c.power}`)));
      return (b, p) => cells.has(`${b}-${p}`) ? WARM : null;
    }
    case 'column-mirrors': {
      const mirrors = findDivergentMirrors(grid, base);
      return (b, p, v) => {
        for (const m of mirrors) {
          if (m.power !== p) continue;
          if (m.type === 'single-axis') {
            if (b === m.centerBase) return AXIS;
            const dist = Math.abs(b - m.centerBase);
            if (dist > 0 && dist <= m.radius) return GOLD_MID;
          } else if (m.type === 'double-axis') {
            if (b === m.centerBase1 || b === m.centerBase2) return AXIS;
            const dist = Math.min(Math.abs(b - m.centerBase1), Math.abs(b - m.centerBase2));
            if (dist > 0 && dist <= m.radius) return GOLD_MID;
          }
        }
        return null;
      };
    }
    case 'column-repeats': {
      const reps = findRepeatingColumnSegments(grid);
      const repMap = {};
      reps.forEach(r => { repMap[r.power] = r; });
      return (b, p) => {
        const rep = repMap[p];
        if (!rep) return null;
        const segIdx = Math.floor((b - 1) / rep.segmentLength);
        return segIdx % 2 === 0 ? GOLD_MID : SILVER;
      };
    }
    case 'column-twins': {
      const twins = findIdenticalColumns(grid);
      const colorMap = {};
      const colors = [GOLD_MID, COOL, SILVER_STRONG];
      twins.forEach((t, i) => t.powers.forEach(p => { colorMap[p] = colors[i % colors.length]; }));
      return (b, p) => colorMap[p] || null;
    }
    case 'column-attractors': {
      const att = findColumnAttractors(grid);
      const attMap = {};
      att.forEach(a => { attMap[a.power] = a.value; });
      return (b, p, v) => {
        if (attMap[p] !== undefined && v === attMap[p]) return DEEP;
        return null;
      };
    }
    case 'column-complements': {
      const comp = findComplementaryColumns(grid, base);
      const colorMap = {};
      comp.forEach(c => { colorMap[c.power1] = GOLD_MID; colorMap[c.power2] = COOL; });
      return (b, p) => colorMap[p] || null;
    }
    case 'column-progressions': {
      const progs = findArithmeticColumnProgressions(grid);
      const cells = new Set();
      progs.forEach(p => {
        for (let i = 0; i < p.length; i++) cells.add(`${p.start + i + 1}-${p.power}`);
      });
      return (b, p) => cells.has(`${b}-${p}`) ? GOLD_DIM : null;
    }
    case 'collapsed-columns':
    case 'collapsed-rows':
      return null;
    default:
      return null;
  }
}

/* ═══════════════════════════════════════════
   Analysis panels
   ═══════════════════════════════════════════ */

function AnalysisPanel({ pattern, grid, base }) {
  const mod = base - 1;

  const A = ({ children }) => <div style={analysisTitleStyle}>{children}</div>;
  const P = ({ children, style: s }) => <p style={{ ...analysisText, ...s }}>{children}</p>;
  const Tag = ({ children }) => <span style={tagStyle}>{children}</span>;
  const Gold = ({ children }) => <span style={{ color: GOLD, fontWeight: 500 }}>{children}</span>;

  switch (pattern) {
    case 'inverse-1':
    case 'inverse-2': {
      const pairs = findMultiplicativeInverses(grid, base);
      const idx = pattern === 'inverse-1' ? 0 : (pairs.length > 1 ? 1 : 0);
      const pair = pairs[idx];
      return (
        <div>
          <A>{pattern === 'inverse-1' ? 'Primary' : 'Secondary'} Inverse Pair — Base {base}</A>
          {pair ? (
            <>
              <P><Gold>Red:</Gold> Base {pair[0]} — [{grid[pair[0] - 1].slice(1, 8).join(', ')}]</P>
              <P><Gold>Blue:</Gold> Base {pair[1]} — [{grid[pair[1] - 1].slice(1, 8).join(', ')}]</P>
              <P>Verification: {pair[0]} × {pair[1]} = {pair[0] * pair[1]} ≡ {(pair[0] * pair[1]) % mod || mod} (mod {mod})</P>
              {pairs.length > 0 && <P>All pairs: {pairs.map(p => `${p[0]}↔${p[1]}`).join(', ')}</P>}
            </>
          ) : <P>No multiplicative inverse pairs exist in this base.</P>}
        </div>
      );
    }
    case 'attractors': {
      const att = findAttractors(grid, base);
      return (
        <div>
          <A>Attractor Bases — Base {base}</A>
          {att.length > 0 ? (
            <>
              <P>Attractors: {att.join(', ')}</P>
              {att.map(b => {
                const seq = grid[b - 1].slice(1, 8);
                return <P key={b}><Gold>Base {b}:</Gold> {seq.join(' → ')} (converges to {seq[seq.length - 1]})</P>;
              })}
            </>
          ) : <P>No attractor bases found.</P>}
        </div>
      );
    }
    case 'cycles': {
      const cycles = grid.map((_, i) => ({ base: i + 1, len: findCycleLength(grid, i) }));
      return (
        <div>
          <A>Cycle Lengths — Base {base}</A>
          <P>Highlighted cells show each base's repeating cycle.</P>
          {cycles.slice(0, Math.min(10, cycles.length)).map(c => {
            const pat = grid[c.base - 1].slice(1, Math.min(c.len + 1, 8));
            return <P key={c.base}><Gold>Base {c.base}:</Gold> length {c.len}{c.len === 1 ? ' (attractor)' : ''}, pattern [{pat.join(', ')}]</P>;
          })}
        </div>
      );
    }
    case 'identity': {
      let count = 0;
      grid.forEach(r => r.slice(1).forEach(v => { if (v === 1) count++; }));
      return (
        <div>
          <A>Identity Element — Base {base}</A>
          <P>Value 1 appears {count} times. These mark positions where base^power ≡ 1 (mod {mod}).</P>
          <P>Base 1 always produces 1 for all powers. Other bases return to 1 based on their cycle length in the multiplicative group.</P>
        </div>
      );
    }
    case 'echoes': {
      const freq = findFrequentValues(grid, base);
      const labels = ['Gold', 'Purple', 'Silver'];
      return (
        <div>
          <A>Frequent Values — Base {base}</A>
          {freq.map((f, i) => <P key={i}><Gold>{labels[i]}: Value {f.value}</Gold> appears {f.count} times</P>)}
        </div>
      );
    }
    case 'diagonals': {
      const diags = findAllDiagonals(grid);
      const labels = ['Gold', 'Blue'];
      return (
        <div>
          <A>Diagonal Chains — Base {base}</A>
          <P>{diags.length} total diagonals found.</P>
          {diags.slice(0, 2).map((d, i) => (
            <P key={i}><Gold>{labels[i]}: Value {d.value}</Gold> repeats {d.count} times ({d.direction})</P>
          ))}
        </div>
      );
    }
    case 'oscillations': {
      const osc = findOscillations(grid);
      return (
        <div>
          <A>Oscillations — Base {base}</A>
          <P>{osc.length} A-B-A-B patterns found. Rows: {osc.filter(o => o.type === 'row').length}, Columns: {osc.filter(o => o.type === 'column').length}.</P>
        </div>
      );
    }
    case 'column-mirrors': {
      const mirrors = findDivergentMirrors(grid, base);
      return (
        <div>
          <A>Divergent Mirrors — Base {base}</A>
          {mirrors.length > 0 ? (
            <>
              <P>{mirrors.length} mirror columns found.</P>
              {mirrors.slice(0, 5).map((m, i) => (
                <P key={i}>
                  <Gold>Column ^{m.power}:</Gold> {m.type === 'single-axis'
                    ? `diverges from base ${m.centerBase}, radius ${m.radius}`
                    : `diverges from bases ${m.centerBase1}–${m.centerBase2}, radius ${m.radius}`
                  }
                </P>
              ))}
              <P style={{ color: 'var(--text-muted)', fontSize: 12 }}>Red = mirror axis · Gold = mirrored values</P>
            </>
          ) : <P>No mirror patterns found.</P>}
        </div>
      );
    }
    case 'column-repeats': {
      const reps = findRepeatingColumnSegments(grid);
      return (
        <div>
          <A>Column Tiling — Base {base}</A>
          {reps.length > 0 ? (
            <>
              {reps.map(r => (
                <P key={r.power}><Gold>Column ^{r.power}:</Gold> [{r.segment.join(', ')}] repeats {r.repeatCount}× (period {r.segmentLength})</P>
              ))}
              <P style={{ fontStyle: 'italic', color: GOLD, opacity: 0.8 }}>This tiling extends to infinity. The segment length divides the group order.</P>
            </>
          ) : <P>No column tiling found.</P>}
        </div>
      );
    }
    case 'column-twins': {
      const twins = findIdenticalColumns(grid);
      return (
        <div>
          <A>Identical Columns — Base {base}</A>
          {twins.length > 0
            ? twins.map((t, i) => <P key={i}>Powers {t.powers.map(p => `^${p}`).join(', ')} produce [{t.data.join(', ')}]</P>)
            : <P>No identical column pairs found.</P>
          }
        </div>
      );
    }
    case 'column-attractors': {
      const att = findColumnAttractors(grid);
      return (
        <div>
          <A>Column Attractors — Base {base}</A>
          {att.length > 0
            ? att.map(a => <P key={a.power}><Gold>Column ^{a.power}:</Gold> value {a.value} appears {a.count} times ({a.percentage}%)</P>)
            : <P>No column attractors found (no value dominates &gt;50% of any column).</P>
          }
        </div>
      );
    }
    case 'column-complements': {
      const comp = findComplementaryColumns(grid, base);
      const typeLabels = {
        'sum-to-modulus': `values sum to ${mod}`,
        'multiplicative-inverse': `products ≡ 1 (mod ${mod})`,
        'constant-difference': 'constant difference'
      };
      return (
        <div>
          <A>Column Complements — Base {base}</A>
          {comp.length > 0
            ? comp.map((c, i) => (
              <P key={i}><Gold>^{c.power1} ↔ ^{c.power2}:</Gold> {typeLabels[c.type]}{c.difference !== undefined ? ` of ${c.difference}` : ''} ({c.strength})</P>
            ))
            : <P>No complementary column pairs found.</P>
          }
        </div>
      );
    }
    case 'column-progressions': {
      const progs = findArithmeticColumnProgressions(grid);
      return (
        <div>
          <A>Column Progressions — Base {base}</A>
          {progs.length > 0 ? (
            <>
              <P>{progs.length} arithmetic progressions found.</P>
              {progs.slice(0, 5).map((p, i) => (
                <P key={i}><Gold>Column ^{p.power}:</Gold> {p.length}-term [{p.sequence.join(', ')}] d={p.difference > 0 ? '+' : ''}{p.difference}</P>
              ))}
            </>
          ) : <P>No arithmetic progressions of length ≥3 found.</P>}
        </div>
      );
    }
    case 'collapsed-columns': {
      const roots = getCollapsedColumnRoots(grid, base);
      const values = roots.map(r => r.digitalRoot);
      return (
        <div>
          <A>Collapsed Column Signature — Base {base}</A>
          <div style={signatureBox}>
            {values.map((r, i) => (
              <div key={i} style={{ fontFamily: 'var(--font-mono)', fontSize: 24, fontWeight: 700, color: GOLD, width: 40, textAlign: 'center' }}>{r}</div>
            ))}
          </div>
          <P>Pattern: {detectSequencePattern(values)}</P>
        </div>
      );
    }
    case 'collapsed-rows': {
      const roots = getCollapsedRowRoots(grid, base);
      const values = roots.map(r => r.digitalRoot);
      return (
        <div>
          <A>Collapsed Row Signature — Base {base}</A>
          <div style={signatureBox}>
            {values.map((r, i) => (
              <div key={i} style={{ fontFamily: 'var(--font-mono)', fontSize: 18, fontWeight: 700, color: GOLD, width: 32, textAlign: 'center' }}>{r}</div>
            ))}
          </div>
          <P>Pattern: {detectSequencePattern(values)}</P>
        </div>
      );
    }
    default: return null;
  }
}

const analysisTitleStyle = {
  fontFamily: 'var(--font-mono)', fontSize: 12, color: GOLD,
  letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: 16, fontWeight: 600
};
const analysisText = { fontSize: 14, color: 'var(--text-muted)', margin: '8px 0', lineHeight: 1.7 };
const tagStyle = {
  fontFamily: 'var(--font-mono)', fontSize: 12, color: GOLD,
  padding: '6px 12px', border: '1px solid rgba(228,187,124,0.25)',
  borderRadius: 4, background: 'rgba(228,187,124,0.05)'
};
const signatureBox = {
  display: 'flex', gap: 8, justifyContent: 'center', margin: '24px 0', flexWrap: 'wrap',
  padding: '20px', background: 'rgba(228,187,124,0.04)',
  border: '1px solid rgba(228,187,124,0.15)', borderRadius: 6
};

/* ═══════════════════════════════════════════
   Main page
   ═══════════════════════════════════════════ */

export default function Discovery() {
  const [base, setBase] = useState(10);
  const [activePattern, setActivePattern] = useState('inverse-1');

  const grid = useMemo(() => buildGrid(base), [base]);
  const highlightFn = useMemo(() => getHighlightFn(activePattern, grid, base), [activePattern, grid, base]);
  const currentPattern = PATTERNS.find(p => p.id === activePattern);

  const groups = {};
  PATTERNS.forEach(p => { if (!groups[p.group]) groups[p.group] = []; groups[p.group].push(p); });

  return (
    <div style={{ flex: 1, paddingBottom: 80 }}>

      {/* Hero */}
      <section style={{
        background: 'linear-gradient(135deg, var(--navy-dark) 0%, var(--navy-mid) 50%, var(--navy-light) 100%)',
        paddingTop: 80, paddingBottom: 80, paddingLeft: 24, paddingRight: 24
      }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 13, color: GOLD,
            letterSpacing: '0.25em', textTransform: 'uppercase', marginBottom: 20, fontWeight: 500
          }}>Where It Started</div>
          <h1 style={{
            fontFamily: 'var(--font-serif)', fontSize: 64, fontWeight: 300,
            color: 'var(--text)', margin: '0 0 16px', letterSpacing: '-1px'
          }}>Initial Discovery</h1>
          <p style={{
            fontFamily: 'var(--font-body)', fontSize: 20, color: GOLD,
            margin: 0, maxWidth: 760, lineHeight: 1.7, opacity: 0.7
          }}>The digital root power table — the observation that launched the programme.</p>
        </div>
      </section>

      {/* Narrative */}
      <section style={{
        background: 'rgba(228,187,124,0.03)',
        borderBottom: '1px solid rgba(228,187,124,0.1)',
        padding: '56px 24px'
      }}>
        <div style={{ maxWidth: 900, margin: '0 auto' }}>
          <p style={{ fontFamily: 'var(--font-body)', fontSize: 17, color: 'var(--text)', lineHeight: 1.9, margin: '0 0 20px' }}>
            Compute the digital root of b<sup>p</sup> for every base and power in a given number system.
            Arrange the results in a table. What emerges is not noise — it is a lattice of
            multiplicative inverse pairs, fixed-point attractors, exact periodic tiling, and
            algebraic symmetries that persist identically across every base system tested.
          </p>
          <p style={{ fontFamily: 'var(--font-body)', fontSize: 17, color: 'var(--text-muted)', lineHeight: 1.9, margin: '0 0 20px' }}>
            This table is the origin of Marithmetics. The patterns observed here —
            particularly the structural triple (137, 107, 103) and its base-invariant behaviour —
            led to the question: if integer geometry carries this much internal structure,
            what happens when you smooth it with an admissible operator?
          </p>
          <p style={{ fontFamily: 'var(--font-body)', fontSize: 17, color: 'var(--text-muted)', lineHeight: 1.9, margin: 0 }}>
            The answer became the framework. Select a base and a pattern below to explore the table yourself.
          </p>
        </div>
      </section>

      {/* Controls */}
      <section style={{
        background: 'rgba(13,17,23,0.97)',
        borderBottom: '1px solid rgba(228,187,124,0.1)',
        padding: '24px 24px', position: 'sticky', top: 64, zIndex: 50,
        backdropFilter: 'blur(12px)'
      }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          {/* Base selector */}
          <div style={{ marginBottom: 16 }}>
            <div style={controlLabel}>Numerical Base (mod {base - 1})</div>
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              {BASE_OPTIONS.map(opt => (
                <button key={opt.value} onClick={() => setBase(opt.value)} style={{
                  fontFamily: 'var(--font-mono)', fontSize: 12, padding: '6px 12px',
                  background: base === opt.value ? 'rgba(228,187,124,0.15)' : 'transparent',
                  border: `1px solid ${base === opt.value ? GOLD : 'var(--border)'}`,
                  borderRadius: 4, cursor: 'pointer',
                  color: base === opt.value ? GOLD : 'var(--text-muted)',
                  transition: 'all 0.15s ease'
                }}>{opt.value}</button>
              ))}
            </div>
          </div>

          {/* Pattern selector grouped */}
          {Object.entries(groups).map(([group, patterns]) => (
            <div key={group} style={{ marginBottom: 8 }}>
              <div style={{ ...controlLabel, fontSize: 10, marginBottom: 4, opacity: 0.5 }}>{group}</div>
              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                {patterns.map(p => (
                  <button key={p.id} onClick={() => setActivePattern(p.id)} style={{
                    fontFamily: 'var(--font-mono)', fontSize: 10, padding: '5px 10px',
                    background: activePattern === p.id ? 'rgba(228,187,124,0.15)' : 'transparent',
                    border: `1px solid ${activePattern === p.id ? GOLD : 'var(--border)'}`,
                    borderRadius: 3, cursor: 'pointer',
                    color: activePattern === p.id ? GOLD : 'var(--text-muted)',
                    transition: 'all 0.15s ease', whiteSpace: 'nowrap'
                  }}>{p.label}</button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Main content */}
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 24px' }}>

        {currentPattern && (
          <div style={{ padding: '32px 0 0', borderBottom: '1px solid rgba(228,187,124,0.08)', marginBottom: 8 }}>
            <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: 32, fontWeight: 400, color: 'var(--text)', margin: '0 0 8px' }}>
              {currentPattern.label}
            </h2>
            <p style={{ fontFamily: 'var(--font-body)', fontSize: 15, color: GOLD, opacity: 0.7, margin: '0 0 24px' }}>
              {currentPattern.desc}
            </p>
          </div>
        )}

        <DRPTTable grid={grid} highlightFn={highlightFn} />

        <div style={{
          background: 'rgba(228,187,124,0.03)',
          border: '1px solid rgba(228,187,124,0.1)',
          borderRadius: 6, padding: 32, marginTop: 16, marginBottom: 64
        }}>
          <AnalysisPanel pattern={activePattern} grid={grid} base={base} />
        </div>

        {/* Base-invariance prompt */}
        <div style={{
          borderLeft: '4px solid var(--gold)', padding: '40px 48px',
          margin: '0 0 64px', background: 'rgba(228,187,124,0.02)'
        }}>
          <h3 style={{
            fontFamily: 'var(--font-serif)', fontSize: 24, fontWeight: 300,
            color: 'var(--text)', margin: '0 0 16px', fontStyle: 'italic'
          }}>Try switching the base.</h3>
          <p style={{ fontFamily: 'var(--font-body)', fontSize: 15, color: 'var(--text-muted)', margin: 0, lineHeight: 1.8 }}>
            The structural relationships — inverse pairs, cycle lengths, tiling periods — transform
            predictably across bases. They are not artifacts of decimal representation. This
            base-gauge invariance is the first testable claim of the framework, formalised in{' '}
            <Link to="/demos/demo-64-base-gauge-invariance-integer-selector" style={{ color: GOLD, textDecoration: 'none' }}>DEMO-64</Link>.
          </p>
        </div>

        {/* CTA */}
        <div style={{ textAlign: 'center', margin: '0 0 80px' }}>
          <div style={{ height: 2, background: 'linear-gradient(90deg, transparent, var(--gold), transparent)', marginBottom: 48 }} />
          <p style={{ fontFamily: 'var(--font-serif)', fontSize: 22, fontWeight: 300, color: 'var(--text)', margin: '0 0 24px' }}>
            From this table to a zero-parameter derivation of physical constants.
          </p>
          <div style={{ display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap' }}>
            <Link to="/demos" style={{ textDecoration: 'none' }}>
              <button style={ctaBtn}>Explore the Demos</button>
            </Link>
            <Link to="/falsification" style={{ textDecoration: 'none' }}>
              <button style={ctaBtnOutline}>How to Break This</button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

const controlLabel = {
  fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)',
  letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 6
};
const ctaBtn = {
  fontFamily: 'var(--font-serif)', fontSize: 16, padding: '14px 40px',
  backgroundColor: GOLD, color: 'var(--bg)', border: 'none',
  borderRadius: 4, cursor: 'pointer', fontWeight: 500, transition: 'all 0.3s ease'
};
const ctaBtnOutline = {
  fontFamily: 'var(--font-serif)', fontSize: 16, padding: '14px 40px',
  backgroundColor: 'transparent', color: GOLD,
  border: `2px solid ${GOLD}`, borderRadius: 4, cursor: 'pointer',
  fontWeight: 500, transition: 'all 0.3s ease'
};

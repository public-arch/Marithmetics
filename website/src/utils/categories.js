export const CATEGORY_LABELS = {
  bridge: 'Bridge',
  controllers: 'Controllers',
  cosmo: 'Cosmology',
  foundations: 'Foundations',
  general_relativity: 'General Relativity',
  infinity: 'Infinity',
  quantum: 'Quantum',
  quantum_gravity: 'Quantum Gravity',
  standard_model: 'Standard Model',
  substrate: 'Substrate'
};

export const CATEGORY_COLORS = {
  bridge: '#e4bb7c',
  controllers: '#60a5fa',
  cosmo: '#a78bfa',
  foundations: '#f472b6',
  general_relativity: '#34d399',
  infinity: '#fb923c',
  quantum: '#22d3ee',
  quantum_gravity: '#c084fc',
  standard_model: '#fbbf24',
  substrate: '#94a3b8'
};

export function getCategoryLabel(category) {
  return CATEGORY_LABELS[category] || category;
}

export function getCategoryColor(category) {
  return CATEGORY_COLORS[category] || '#e4bb7c';
}

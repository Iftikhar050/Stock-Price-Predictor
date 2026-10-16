const UNITS = [
  { value: 1e12, suffix: "T" },
  { value: 1e9, suffix: "B" },
  { value: 1e6, suffix: "M" },
  { value: 1e3, suffix: "K" },
];

function scale(value, minUnit, digits) {
  const abs = Math.abs(value);
  for (const u of UNITS) {
    if (u.value < minUnit) break;
    if (abs >= u.value) return `${(value / u.value).toFixed(digits)}${u.suffix}`;
  }
  return null;
}

// Millions and up in M/B/T; smaller values keep their full digits.
export function formatNumber(value, digits = 2) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return scale(value, 1e6, digits) ?? value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

// Also shortens thousands to K - for chart axes and tight stat tiles.
export function formatCompact(value, digits = 2) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return scale(value, 1e3, digits) ?? value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

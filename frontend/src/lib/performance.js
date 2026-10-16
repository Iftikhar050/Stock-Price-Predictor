const WINDOWS = [
  { key: "1D", days: 1 },
  { key: "1W", days: 7 },
  { key: "1M", days: 30 },
  { key: "3M", days: 90 },
  { key: "6M", days: 182 },
  { key: "1Y", days: 365 },
];

export const PERFORMANCE_WINDOW_KEYS = [...WINDOWS.map((w) => w.key), "YTD"];

// series must be ascending by date: [{ date: "YYYY-MM-DD", value: number }]
function valueAtOrBefore(series, targetDate) {
  let result = series[0]?.value ?? null;
  for (const point of series) {
    if (new Date(point.date) <= targetDate) result = point.value;
    else break;
  }
  return result;
}

// Returns { "1D": pct, "1W": pct, ..., "YTD": pct } computed against the last
// point in `series`. Missing/insufficient history yields null for that window,
// not a fabricated 0.
export function computeReturns(series) {
  const out = {};
  if (!series || series.length === 0) {
    PERFORMANCE_WINDOW_KEYS.forEach((k) => { out[k] = null; });
    return out;
  }

  const latest = series[series.length - 1];
  const latestValue = latest.value;
  const latestDate = new Date(latest.date);

  for (const w of WINDOWS) {
    const cutoff = new Date(latestDate);
    cutoff.setDate(cutoff.getDate() - w.days);
    const past = valueAtOrBefore(series, cutoff);
    out[w.key] = past ? ((latestValue - past) / past) * 100 : null;
  }

  const yearStart = new Date(latestDate.getFullYear(), 0, 1);
  const ytdPast = valueAtOrBefore(series, yearStart);
  out.YTD = ytdPast ? ((latestValue - ytdPast) / ytdPast) * 100 : null;

  return out;
}

// Rebases a price series to "% change since the first point" so tickers on
// wildly different price scales (e.g. Rs. 10 vs Rs. 500) can share one chart
// axis. series must be ascending by date: [{ date, value }].
export function rebaseToPercent(series) {
  if (!series || series.length === 0) return [];
  const base = series[0].value;
  if (!base) return [];
  return series.map((p) => ({ date: p.date, value: ((p.value - base) / base) * 100 }));
}

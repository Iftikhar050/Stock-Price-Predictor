export const RANGE_OPTIONS = ["7D", "30D", "90D", "1Y"];

const RANGE_DAYS = { "7D": 7, "30D": 30, "90D": 90, "1Y": 365 };

// Trims an ascending [{ date: "YYYY-MM-DD", ... }] series to the trailing
// window ending at its own last point (not "today" - history can lag).
export function filterHistoryByRange(history, range) {
  if (!history || history.length === 0) return [];
  const days = RANGE_DAYS[range];
  if (!days) return history;
  const lastDate = new Date(history[history.length - 1].date);
  const cutoff = new Date(lastDate);
  cutoff.setDate(cutoff.getDate() - days);
  return history.filter((p) => new Date(p.date) >= cutoff);
}

// Formats a "YYYY-MM-DD" tick label sensibly for the given range string -
// falls back to month/day for anything not explicitly a 1Y/5Y-style range.
export function formatChartDate(dateStr, range) {
  if (!dateStr) return "";
  const [year, month, day] = dateStr.split("-");
  if (!month) return dateStr;
  const dateObj = new Date(year, parseInt(month, 10) - 1, day);
  if (range === "5Y") return year;
  if (range === "1Y") return dateObj.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
  return dateObj.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// Merges { seriesKey: [{date, value}] } series (already ascending) into one
// array keyed by date for a shared-axis recharts dataset:
// [{ date, seriesKey1: value, seriesKey2: value }].
export function mergeSeriesByDate(seriesByKey) {
  const maps = {};
  const dateSet = new Set();
  for (const [key, series] of Object.entries(seriesByKey)) {
    maps[key] = new Map(series.map((p) => [p.date, p.value]));
    series.forEach((p) => dateSet.add(p.date));
  }
  const dates = [...dateSet].sort();
  return dates.map((date) => {
    const row = { date };
    for (const key of Object.keys(maps)) {
      const v = maps[key].get(date);
      if (v !== undefined) row[key] = v;
    }
    return row;
  });
}

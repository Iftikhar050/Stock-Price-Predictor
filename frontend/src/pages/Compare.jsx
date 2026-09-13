import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import Select from "react-select";
import {
  LineChart, Line, BarChart, Bar, Cell, LabelList, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { API_BASE_URL } from "../config";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "../components/ui/table";
import { Skeleton } from "../components/ui/skeleton";
import { selectStyles } from "../lib/reactSelectStyles";
import { rebaseToPercent } from "../lib/performance";
import { formatChartDate, mergeSeriesByDate } from "../lib/dateRange";
import { CHART_COLORS } from "../lib/chartColors";

const METRIC_ROWS = [
  { key: "sector", label: "Sector" },
  { key: "price", label: "Price", fmt: (v) => v?.toFixed(2) },
  { key: "change_percent", label: "Change %", fmt: (v) => (v != null ? `${v.toFixed(2)}%` : "—") },
  { key: "market_cap", label: "Market Cap", fmt: (v) => (v != null ? v.toLocaleString() : "—") },
  { key: "pe_ratio", label: "P/E", fmt: (v) => v?.toFixed(2) },
  { key: "pb_ratio", label: "P/B", fmt: (v) => v?.toFixed(2) },
  { key: "dividend_yield", label: "Dividend Yield", fmt: (v) => (v != null ? `${(v * 100).toFixed(2)}%` : "—") },
  { key: "roe", label: "ROE", fmt: (v) => (v != null ? `${(v * 100).toFixed(2)}%` : "—") },
];

const RANGES = ["30D", "90D", "1Y", "5Y"];

const RATIO_CHART_METRICS = [
  { key: "market_cap", label: "Market Cap", formatter: (v) => `Rs. ${fmtCompact(v)}` },
  { key: "pe_ratio", label: "P/E Ratio", formatter: (v) => v.toFixed(1) },
  { key: "pb_ratio", label: "P/B Ratio", formatter: (v) => v.toFixed(1) },
  { key: "dividend_yield", label: "Dividend Yield", formatter: (v) => `${v.toFixed(1)}%`, scale: 100 },
  { key: "roe", label: "Return on Equity", formatter: (v) => `${v.toFixed(1)}%`, scale: 100 },
];

function fmtCompact(value) {
  if (typeof value !== "number") return "—";
  if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toLocaleString();
}

function PricePerformanceChart({ selected, histories, historiesLoading, range, setRange, colorFor }) {
  const chartData = useMemo(() => {
    const rebased = {};
    for (const [ticker, series] of Object.entries(histories)) {
      rebased[ticker] = rebaseToPercent(series);
    }
    const merged = mergeSeriesByDate(rebased);

    // The line/axis stay % (necessary to compare stocks on wildly different
    // price scales on one chart), but the tooltip should show the real
    // price people actually care about - stash it alongside under a
    // "__price" key per ticker so the Tooltip formatter can look it up.
    const rawByDate = {};
    for (const [ticker, series] of Object.entries(histories)) {
      rawByDate[ticker] = new Map(series.map((p) => [p.date, p.value]));
    }
    return merged.map((row) => {
      const out = { ...row };
      for (const ticker of Object.keys(rawByDate)) {
        const v = rawByDate[ticker].get(row.date);
        if (v !== undefined) out[`${ticker}__price`] = v;
      }
      return out;
    });
  }, [histories]);

  return (
    <Card className="shadow-sm">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Price Performance</CardTitle>
        <div className="flex gap-1 rounded-lg bg-muted p-1">
          {RANGES.map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={`rounded-md px-2.5 py-1 text-xs font-semibold transition-colors ${range === r ? "bg-card text-foreground shadow-sm" : "text-muted-foreground"}`}
            >
              {r}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent>
        <p className="mb-2 text-xs text-muted-foreground">Lines show % change since the start of the range (puts different price scales on one comparable axis) — hover a point to see the real price.</p>
        {historiesLoading ? (
          <Skeleton className="h-72 w-full" />
        ) : (
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="var(--border)" strokeOpacity={0.6} vertical={false} strokeDasharray="4 4" />
                <XAxis
                  dataKey="date"
                  stroke="var(--muted-foreground)"
                  tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                  tickFormatter={(v) => formatChartDate(v, range)}
                  minTickGap={40}
                />
                <YAxis
                  stroke="var(--muted-foreground)"
                  tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                  tickFormatter={(v) => `${v.toFixed(0)}%`}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", borderRadius: "8px", color: "var(--foreground)" }}
                  formatter={(value, name, props) => {
                    const price = props.payload?.[`${props.dataKey}__price`];
                    return [price != null ? `Rs. ${price.toFixed(2)}` : "—", name];
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {selected.map((s) => (
                  <Line
                    key={s.value}
                    type="monotone"
                    dataKey={s.value}
                    name={s.value}
                    stroke={colorFor(s.value)}
                    strokeWidth={2}
                    dot={false}
                    connectNulls
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ColoredTick({ x, y, payload, colorFor }) {
  return (
    <text x={x} y={y + 14} textAnchor="middle" fontSize={12} fontWeight={700} fill={colorFor(payload.value)}>
      {payload.value}
    </text>
  );
}

function RatioComparisonCharts({ columns, colorFor }) {
  const charts = useMemo(() => RATIO_CHART_METRICS.map((m) => {
    // market_cap of exactly 0 means missing shares-outstanding data upstream,
    // not a real zero-valued company - exclude rather than plot a false bar.
    const data = columns
      .filter((c) => (m.key === "market_cap" ? c[m.key] > 0 : c[m.key] != null))
      .map((c) => ({ ticker: c.ticker, value: m.scale ? c[m.key] * m.scale : c[m.key] }));
    return { ...m, data };
  }).filter((c) => c.data.length > 0), [columns]);

  if (charts.length === 0) return null;

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Ratio Comparison</CardTitle>
        <p className="text-xs text-muted-foreground">Key valuation and profitability metrics, side by side.</p>
      </CardHeader>
      <CardContent className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {charts.map((c) => (
          <div key={c.key} className="rounded-xl border border-border bg-muted/30 p-4">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">{c.label}</p>
            <div className="h-52 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={c.data} margin={{ top: 24, right: 8, left: 8, bottom: 0 }} barCategoryGap="30%">
                  <CartesianGrid stroke="var(--border)" strokeOpacity={0.5} vertical={false} strokeDasharray="4 4" />
                  <XAxis
                    dataKey="ticker"
                    tick={(props) => <ColoredTick {...props} colorFor={colorFor} />}
                    tickLine={false}
                    axisLine={{ stroke: "var(--border)" }}
                  />
                  <YAxis hide domain={[0, (max) => max * 1.2]} />
                  <Tooltip
                    cursor={{ fill: "var(--muted)" }}
                    contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", borderRadius: "8px", color: "var(--foreground)" }}
                    formatter={(v) => c.formatter(v)}
                  />
                  <Bar dataKey="value" radius={[8, 8, 0, 0]} maxBarSize={56} isAnimationActive={false}>
                    {c.data.map((d) => (
                      <Cell key={d.ticker} fill={colorFor(d.ticker)} />
                    ))}
                    <LabelList
                      dataKey="value"
                      position="top"
                      formatter={c.formatter}
                      style={{ fontSize: 11, fontWeight: 600, fill: "var(--foreground)" }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export function Compare() {
  const [tickerOptions, setTickerOptions] = useState([]);
  const [selected, setSelected] = useState([]);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [histories, setHistories] = useState({});
  const [historiesLoading, setHistoriesLoading] = useState(false);
  const [range, setRange] = useState("1Y");

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/api/tickers`)
      .then((res) => setTickerOptions(res.data.map((t) => ({ value: t.ticker, label: `${t.ticker} — ${t.name}` }))))
      .catch((err) => console.error("Failed to fetch tickers", err));
  }, []);

  useEffect(() => {
    if (selected.length < 2) {
      setResults(null);
      setError(null);
      return;
    }
    if (selected.length > 5) {
      setError("Pick at most 5 companies.");
      return;
    }
    setError(null);
    axios
      .get(`${API_BASE_URL}/api/compare`, { params: { tickers: selected.map((s) => s.value).join(",") } })
      .then((res) => setResults(res.data.results))
      .catch((err) => setError(err.response?.data?.detail || "Failed to compare tickers."));
  }, [selected]);

  useEffect(() => {
    if (selected.length < 2 || selected.length > 5) {
      setHistories({});
      return;
    }
    setHistoriesLoading(true);
    Promise.all(
      selected.map((s) =>
        axios
          .get(`${API_BASE_URL}/api/company/${s.value}/history`, { params: { range } })
          .then((res) => [s.value, (res.data.history ?? []).map((h) => ({ date: h.date, value: h.close }))])
          .catch(() => [s.value, []])
      )
    )
      .then((pairs) => setHistories(Object.fromEntries(pairs)))
      .finally(() => setHistoriesLoading(false));
  }, [selected, range]);

  const columns = useMemo(() => results ?? [], [results]);

  const colorMap = useMemo(() => {
    const map = {};
    selected.forEach((s, idx) => { map[s.value] = CHART_COLORS[idx % CHART_COLORS.length]; });
    return map;
  }, [selected]);
  const colorFor = (ticker) => colorMap[ticker] ?? "var(--muted-foreground)";

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold">Compare</h1>
        <p className="text-sm text-muted-foreground">Pick 2–5 companies for a side-by-side comparison.</p>
      </div>

      <Select
        isMulti
        options={tickerOptions}
        value={selected}
        onChange={setSelected}
        styles={selectStyles}
        placeholder="Select companies to compare..."
      />

      {error && <p className="text-sm text-negative">{error}</p>}

      {columns.length > 0 && (
        <>
          <PricePerformanceChart
            selected={selected}
            histories={histories}
            historiesLoading={historiesLoading}
            range={range}
            setRange={setRange}
            colorFor={colorFor}
          />

          <RatioComparisonCharts columns={columns} colorFor={colorFor} />

          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle>Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Metric</TableHead>
                    {columns.map((c) => (
                      <TableHead key={c.ticker} className="text-right text-sm font-bold" style={{ color: colorFor(c.ticker) }}>{c.ticker}</TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {METRIC_ROWS.map((row) => (
                    <TableRow key={row.key}>
                      <TableCell className="font-medium text-muted-foreground">{row.label}</TableCell>
                      {columns.map((c) => (
                        <TableCell key={c.ticker} className="text-right tabular-nums">
                          {row.fmt ? row.fmt(c[row.key]) : (c[row.key] ?? "—")}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

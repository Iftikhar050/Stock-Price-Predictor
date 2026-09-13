import React, { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";
import { FileDown } from "lucide-react";
import {
  ComposedChart, BarChart, Area, Line, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { API_BASE_URL } from "../config";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell, SortableTableHead } from "../components/ui/table";
import { Skeleton } from "../components/ui/skeleton";
import { SignalBadge } from "../components/SignalBadge";
import { CompanyLogo } from "../components/CompanyLogo";
import { ScreenerTable } from "../components/ScreenerTable";
import { computeSignal } from "../lib/signal";
import { computeReturns, PERFORMANCE_WINDOW_KEYS } from "../lib/performance";
import { useSort } from "../lib/useSort";
import { usePagination } from "../lib/usePagination";
import { PaginationControls } from "../components/ui/pagination";

const MODELS = [
  { key: "RF", label: "Random Forest" },
  { key: "LR", label: "Linear Regression" },
  { key: "XGB", label: "XGBoost" },
  { key: "LSTM", label: "Deep Learning" },
];
const RANGES = ["7D", "30D", "90D", "1Y", "5Y"];
const PREDICTED_LINE_COLOR = "#f59e0b";

const SECTIONS = [
  { id: "overview", label: "Overview" },
  { id: "financials", label: "Financials" },
  { id: "ratios", label: "Ratios" },
  { id: "dividends", label: "Dividends" },
  { id: "peers", label: "Peers" },
  { id: "events", label: "Events" },
  { id: "reports", label: "Reports" },
  { id: "profile", label: "Profile" },
];

function SectionNav({ activeId }) {
  return (
    <div className="sticky top-0 z-10 -mx-4 mb-2 border-b border-border bg-background/95 px-4 py-2 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <nav className="flex gap-1 overflow-x-auto rounded-lg bg-muted p-1 text-muted-foreground">
        {SECTIONS.map((s) => (
          <a
            key={s.id}
            href={`#${s.id}`}
            className={`shrink-0 whitespace-nowrap rounded-md px-3 py-1 text-sm font-medium transition-colors ${
              activeId === s.id ? "bg-card text-foreground shadow-sm" : "hover:text-foreground"
            }`}
          >
            {s.label}
          </a>
        ))}
      </nav>
    </div>
  );
}

function isMarketClosedNow() {
  const now = new Date();
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "Asia/Karachi", hour12: false, hour: "numeric", weekday: "short",
  });
  let hour = 0, weekday = "";
  formatter.formatToParts(now).forEach((p) => {
    if (p.type === "hour") hour = parseInt(p.value, 10);
    if (p.type === "weekday") weekday = p.value;
  });
  return weekday === "Sat" || weekday === "Sun" || hour >= 16;
}

function PriceCard({ data, liveData, marketClosed }) {
  const currentPrice = data.historical_data[data.historical_data.length - 1]?.close ?? data.current_price;
  const previousPrice = data.historical_data[data.historical_data.length - 2]?.close ?? currentPrice;
  const dailyChange = currentPrice - previousPrice;
  const dailyChangePercent = previousPrice ? (dailyChange / previousPrice) * 100 : 0;

  const displayPrice = liveData ? liveData.price : currentPrice;
  const displayChange = liveData ? liveData.change : dailyChange;
  const displayPercent = liveData ? liveData.change_percent : dailyChangePercent;

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 normal-case text-sm font-semibold text-muted-foreground">
          <span className={`inline-block h-2 w-2 rounded-full ${liveData && !marketClosed ? "animate-pulse bg-positive" : "bg-muted-foreground"}`} />
          {liveData && !marketClosed ? "Live Price" : marketClosed ? "Market Closed" : "Current Price"}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-4xl font-bold tabular-nums">Rs. {displayPrice.toFixed(2)}</div>
        <div className={`mt-1 flex items-center gap-1 text-lg font-semibold tabular-nums ${displayChange >= 0 ? "text-positive" : "text-negative"}`}>
          {displayChange >= 0 ? "▲" : "▼"} {Math.abs(displayChange).toFixed(2)} ({displayPercent.toFixed(2)}%)
        </div>
      </CardContent>
    </Card>
  );
}

function PredictionRangeBar({ min, max, current, target }) {
  const range = max - min || 1;
  const clampPct = (v) => Math.min(100, Math.max(0, ((v - min) / range) * 100));
  const currentPct = clampPct(current);
  const targetPct = clampPct(target);

  return (
    <div className="mt-4">
      <div className="mb-1.5 flex items-center justify-between text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
        <span className="tabular-nums">Rs. {min.toFixed(2)}</span>
        <span>Prediction Range</span>
        <span className="tabular-nums">Rs. {max.toFixed(2)}</span>
      </div>
      <div className="relative h-2 rounded-full bg-gradient-to-r from-negative/30 via-muted to-positive/30">
        <div
          className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-card bg-foreground"
          style={{ left: `${currentPct}%` }}
          title={`Current: Rs. ${current.toFixed(2)}`}
        />
        <div
          className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-card bg-primary"
          style={{ left: `${targetPct}%` }}
          title={`Target: Rs. ${target.toFixed(2)}`}
        />
      </div>
      <div className="mt-2 flex justify-center gap-4 text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-foreground" /> Current <span className="tabular-nums font-medium text-foreground">Rs. {current.toFixed(2)}</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-primary" /> Target <span className="tabular-nums font-medium text-foreground">Rs. {target.toFixed(2)}</span>
        </span>
      </div>
    </div>
  );
}

function PredictionCard({ data, modelType, setModelType }) {
  const activePrediction = {
    RF: data.rf_predicted_price, LR: data.lr_predicted_price, XGB: data.xgb_predicted_price, LSTM: data.lstm_predicted_price,
  }[modelType];
  const diff = activePrediction - data.current_price;
  const signal = computeSignal(data);

  return (
    <Card className="border-primary/30 shadow-sm">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>AI Prediction</CardTitle>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Experimental — the underlying model's directional accuracy has not been validated above chance. Not investment advice.
          </p>
        </div>
        <SignalBadge prediction={data} />
      </CardHeader>
      <CardContent>
        <div className="mb-3 flex flex-wrap gap-1">
          {MODELS.map((m) => (
            <button
              key={m.key}
              onClick={() => setModelType(m.key)}
              className={`rounded-full px-2.5 py-1 text-xs font-semibold transition-colors ${
                modelType === m.key ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground hover:bg-accent"
              }`}
            >
              {m.key}
            </button>
          ))}
        </div>
        <div className="text-3xl font-bold tabular-nums">Rs. {activePrediction.toFixed(2)}</div>
        <div className={`mt-1 text-sm font-medium tabular-nums ${diff >= 0 ? "text-positive" : "text-negative"}`}>
          {diff >= 0 ? "Target increase" : "Target decrease"} of Rs. {Math.abs(diff).toFixed(2)}
        </div>

        <PredictionRangeBar min={data.ensemble_min} max={data.ensemble_max} current={data.current_price} target={activePrediction} />

        <div className="mt-4">
          <div className="flex justify-between text-xs font-semibold uppercase text-muted-foreground">
            <span>Model Agreement</span>
            <span className="tabular-nums">{data.model_agreement_score}%</span>
          </div>
          <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={`h-full rounded-full ${data.model_agreement_score >= 80 ? "bg-positive" : data.model_agreement_score >= 60 ? "bg-primary" : "bg-negative"}`}
              style={{ width: `${Math.min(100, Math.max(0, data.model_agreement_score))}%` }}
            />
          </div>
        </div>
        {signal && (
          <p className="mt-3 text-xs text-muted-foreground">
            Ensemble average implies a {signal.pctChange >= 0 ? "+" : ""}{signal.pctChange.toFixed(2)}% move.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function SentimentCard({ sentiment }) {
  const label = sentiment > 0.05 ? "Bullish" : sentiment < -0.05 ? "Bearish" : "Neutral";
  return (
    <Card className="shadow-sm">
      <CardHeader><CardTitle>Market Sentiment</CardTitle></CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{label}</div>
        <div className="mb-3 text-sm text-muted-foreground tabular-nums">Score: {sentiment.toFixed(2)}</div>
        <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted">
          <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
          <div
            className={`absolute inset-y-0 ${sentiment >= 0 ? "bg-positive" : "bg-negative"}`}
            style={{
              width: `${Math.min(50, Math.abs(sentiment) * 50)}%`,
              [sentiment >= 0 ? "left" : "right"]: "50%",
            }}
          />
        </div>
      </CardContent>
    </Card>
  );
}

function DividendCard({ dividend, currentPrice }) {
  if (!dividend) return null;
  return (
    <Card className="shadow-sm">
      <CardHeader><CardTitle>Recent Dividend</CardTitle></CardHeader>
      <CardContent>
        <div className="text-2xl font-bold tabular-nums">Rs. {dividend.amount.toFixed(2)}</div>
        <div className="mb-3 text-sm text-muted-foreground tabular-nums">Yield: {((dividend.amount / currentPrice) * 100).toFixed(2)}%</div>
        <div className="flex items-center justify-between rounded-lg bg-muted px-3 py-2 text-xs font-semibold">
          <span className="text-muted-foreground">Ex-Date</span>
          <span>{new Date(dividend.ex_date).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}</span>
        </div>
      </CardContent>
    </Card>
  );
}

function PriceChart({ data, modelType, timeRange, setTimeRange }) {
  const chartData = useMemo(() => {
    let history = [...data.historical_data];
    const lastDate = new Date(history[history.length - 1].date);
    const cutoff = new Date(lastDate);
    if (timeRange === "7D") cutoff.setDate(cutoff.getDate() - 7);
    if (timeRange === "30D") cutoff.setMonth(cutoff.getMonth() - 1);
    if (timeRange === "90D") cutoff.setMonth(cutoff.getMonth() - 3);
    if (timeRange === "1Y") cutoff.setFullYear(cutoff.getFullYear() - 1);
    if (timeRange === "5Y") cutoff.setFullYear(cutoff.getFullYear() - 5);
    history = history.filter((row) => new Date(row.date) >= cutoff);

    const predKey = { RF: "rf_pred", LR: "lr_pred", XGB: "xgb_pred", LSTM: "lstm_pred" }[modelType];
    history = history.map((row) => ({ ...row, predictedClose: row[predKey] }));

    const predictedPrice = { RF: data.rf_predicted_price, LR: data.lr_predicted_price, XGB: data.xgb_predicted_price, LSTM: data.lstm_predicted_price }[modelType];
    const nextDate = new Date(lastDate);
    nextDate.setDate(lastDate.getDate() + 1);
    history.push({ date: nextDate.toISOString().split("T")[0], predictedClose: predictedPrice });
    return history;
  }, [data, modelType, timeRange]);

  const formatXAxisDate = (tickStr) => {
    if (!tickStr) return "";
    const [year, month, day] = tickStr.split("-");
    if (!month) return tickStr;
    const dateObj = new Date(year, parseInt(month, 10) - 1, day);
    if (timeRange === "5Y") return year;
    if (timeRange === "1Y") return dateObj.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
    return dateObj.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  };
  const tickGap = { "7D": 0, "30D": 10, "90D": 20, "1Y": 40 }[timeRange] ?? 80;

  return (
    <Card className="flex h-[420px] flex-col shadow-sm">
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex gap-1 rounded-lg bg-muted p-1">
          {RANGES.map((r) => (
            <button
              key={r}
              onClick={() => setTimeRange(r)}
              className={`rounded-md px-2.5 py-1 text-xs font-semibold transition-colors ${timeRange === r ? "bg-card text-foreground shadow-sm" : "text-muted-foreground"}`}
            >
              {r}
            </button>
          ))}
        </div>
        <span className="text-xs text-muted-foreground">As of {data.latest_date}</span>
      </CardHeader>
      <CardContent className="flex-1 pb-4">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 0, left: 10, bottom: 0 }}>
            <defs>
              <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--primary)" stopOpacity={0.4} />
                <stop offset="60%" stopColor="var(--primary)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="var(--border)" strokeOpacity={0.6} vertical={false} strokeDasharray="4 4" />
            <XAxis dataKey="date" stroke="var(--muted-foreground)" tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} tickFormatter={formatXAxisDate} minTickGap={tickGap} />
            <YAxis domain={["auto", "auto"]} orientation="right" stroke="var(--muted-foreground)" tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} />
            <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", borderRadius: "8px", color: "var(--foreground)" }} />
            <ReferenceLine x={data.latest_date} stroke="var(--muted-foreground)" strokeDasharray="3 3" label={{ position: "insideTopLeft", value: "Today", fill: "var(--muted-foreground)", fontSize: 11 }} />
            <Area type="linear" dataKey="close" name="Actual Price" stroke="var(--primary)" strokeWidth={2} fillOpacity={1} fill="url(#colorClose)" />
            <Line type="linear" dataKey="predictedClose" name="Predicted Price" stroke={PREDICTED_LINE_COLOR} strokeWidth={2} strokeDasharray="4 4" dot={false} connectNulls />
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function fmtCompact(value) {
  if (typeof value !== "number") return "—";
  if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toLocaleString();
}

function SnapshotStat({ label, value }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-0.5 text-sm font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function CompanySnapshot({ snapshot }) {
  if (!snapshot) return null;
  return (
    <Card className="shadow-sm">
      <CardHeader><CardTitle>Company Snapshot</CardTitle></CardHeader>
      <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
        <SnapshotStat label="52W High" value={snapshot.week_52_high != null ? `Rs. ${snapshot.week_52_high.toFixed(2)}` : "—"} />
        <SnapshotStat label="52W Low" value={snapshot.week_52_low != null ? `Rs. ${snapshot.week_52_low.toFixed(2)}` : "—"} />
        <SnapshotStat label="Market Cap" value={snapshot.market_cap != null ? `Rs. ${fmtCompact(snapshot.market_cap)}` : "—"} />
        <SnapshotStat label="Shares Outstanding" value={fmtCompact(snapshot.shares_outstanding)} />
        <SnapshotStat label="Free Float" value={snapshot.free_float_pct != null ? `${(snapshot.free_float_pct * 100).toFixed(1)}%` : "—"} />
        <SnapshotStat label="EPS (TTM)" value={snapshot.eps_trailing != null ? `Rs. ${snapshot.eps_trailing.toFixed(2)}` : "—"} />
      </CardContent>
    </Card>
  );
}

function InsiderActivityCard({ snapshot }) {
  if (!snapshot) return null;
  const buy = snapshot.insider_buy_shares_30d ?? 0;
  const sell = snapshot.insider_sell_shares_30d ?? 0;
  const net = snapshot.insider_net_flow_30d ?? 0;
  if (!buy && !sell && !net) return null; // nothing to report, don't show a misleadingly-empty card
  return (
    <Card className="shadow-sm">
      <CardHeader><CardTitle>Insider Activity (30D)</CardTitle></CardHeader>
      <CardContent className="grid grid-cols-3 gap-3 text-sm">
        <div>
          <div className="text-xs text-muted-foreground">Bought</div>
          <div className="font-semibold tabular-nums text-positive">{fmtCompact(buy)}</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Sold</div>
          <div className="font-semibold tabular-nums text-negative">{fmtCompact(sell)}</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Net Flow</div>
          <div className={`font-semibold tabular-nums ${net >= 0 ? "text-positive" : "text-negative"}`}>{fmtCompact(net)}</div>
        </div>
      </CardContent>
      <CardContent className="pt-0">
        <p className="text-xs text-muted-foreground">Aggregate director/officer share activity over the trailing 30 days — not a per-transaction record.</p>
      </CardContent>
    </Card>
  );
}

function PerformanceRow({ stockReturns, benchmarkReturns, benchmarkName }) {
  return (
    <Card className="shadow-sm">
      <CardHeader><CardTitle>Performance</CardTitle></CardHeader>
      <CardContent>
        <PerformanceChart stockReturns={stockReturns} benchmarkReturns={benchmarkReturns} benchmarkName={benchmarkName} />
        <div className="mt-4">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead></TableHead>
              {PERFORMANCE_WINDOW_KEYS.map((w) => <TableHead key={w} className="text-right">{w}</TableHead>)}
            </TableRow>
          </TableHeader>
          <TableBody>
            <TableRow>
              <TableCell className="font-medium">This Stock</TableCell>
              {PERFORMANCE_WINDOW_KEYS.map((w) => {
                const v = stockReturns[w];
                return (
                  <TableCell key={w} className={`text-right tabular-nums ${v == null ? "text-muted-foreground" : v >= 0 ? "text-positive" : "text-negative"}`}>
                    {v != null ? `${v >= 0 ? "+" : ""}${v.toFixed(2)}%` : "—"}
                  </TableCell>
                );
              })}
            </TableRow>
            <TableRow>
              <TableCell className="font-medium text-muted-foreground">{benchmarkName}</TableCell>
              {PERFORMANCE_WINDOW_KEYS.map((w) => {
                const v = benchmarkReturns[w];
                return (
                  <TableCell key={w} className={`text-right tabular-nums ${v == null ? "text-muted-foreground" : v >= 0 ? "text-positive" : "text-negative"}`}>
                    {v != null ? `${v >= 0 ? "+" : ""}${v.toFixed(2)}%` : "—"}
                  </TableCell>
                );
              })}
            </TableRow>
          </TableBody>
        </Table>
        </div>
      </CardContent>
    </Card>
  );
}

function FiftyTwoWeekRangeBar({ low, high, current }) {
  if (low == null || high == null || current == null) return null;
  const range = high - low || 1;
  const pct = Math.min(100, Math.max(0, ((current - low) / range) * 100));
  return (
    <Card className="shadow-sm">
      <CardHeader><CardTitle>52-Week Range</CardTitle></CardHeader>
      <CardContent>
        <div className="relative h-2 rounded-full bg-gradient-to-r from-negative/30 via-muted to-positive/30">
          <div
            className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-card bg-foreground"
            style={{ left: `${pct}%` }}
            title={`Current: Rs. ${current.toFixed(2)}`}
          />
        </div>
        <div className="mt-1.5 flex justify-between text-xs text-muted-foreground">
          <span className="tabular-nums">Rs. {low.toFixed(2)}</span>
          <span className="font-semibold text-foreground tabular-nums">Rs. {current.toFixed(2)}</span>
          <span className="tabular-nums">Rs. {high.toFixed(2)}</span>
        </div>
      </CardContent>
    </Card>
  );
}

function SectorRatioChart({ label, companyValue, sectorValue }) {
  const chartData = [{ name: label, Company: companyValue, "Sector Avg": sectorValue }];
  return (
    <div className="h-28 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <XAxis type="number" hide />
          <YAxis type="category" dataKey="name" hide />
          <Tooltip
            cursor={{ fill: "var(--muted)" }}
            contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", borderRadius: "8px", color: "var(--foreground)" }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="Company" fill="var(--primary)" radius={[0, 4, 4, 0]} barSize={16} />
          <Bar dataKey="Sector Avg" fill="var(--muted-foreground)" radius={[0, 4, 4, 0]} barSize={16} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

const PEER_CHART_LIMIT = 7;

function PeerComparisonChart({ ticker, label, formatter = (v) => v.toFixed(2), data }) {
  if (!data || data.length <= 1) return null;
  return (
    <div>
      <p className="mb-1 text-xs text-muted-foreground">{label}</p>
      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--border)" strokeOpacity={0.6} vertical={false} strokeDasharray="4 4" />
            <XAxis dataKey="ticker" stroke="var(--muted-foreground)" tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} />
            <YAxis stroke="var(--muted-foreground)" tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} tickFormatter={formatter} />
            <Tooltip
              cursor={{ fill: "var(--muted)" }}
              contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", borderRadius: "8px", color: "var(--foreground)" }}
              formatter={(v) => formatter(v)}
            />
            <Bar dataKey="value" radius={[3, 3, 0, 0]}>
              {data.map((d) => (
                <Cell key={d.ticker} fill={d.ticker === ticker ? "var(--primary)" : "var(--muted-foreground)"} fillOpacity={d.ticker === ticker ? 1 : 0.5} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const PEER_RATIO_METRICS = [
  { key: "market_cap", label: "Market Cap", formatter: (v) => `Rs. ${fmtCompact(v)}` },
  { key: "pe_ratio", label: "P/E Ratio", formatter: (v) => v.toFixed(1) },
  { key: "pb_ratio", label: "P/B Ratio", formatter: (v) => v.toFixed(1) },
  { key: "dividend_yield", label: "Dividend Yield", formatter: (v) => `${v.toFixed(1)}%`, scale: 100 },
  { key: "roe", label: "Return on Equity", formatter: (v) => `${v.toFixed(1)}%`, scale: 100 },
];

function PeerRatioComparison({ ticker, snapshot, peers }) {
  if (!snapshot || !peers || peers.length === 0) return null;

  const selfRow = { ticker, market_cap: snapshot.market_cap, pe_ratio: snapshot.pe_ratio, pb_ratio: snapshot.pb_ratio, dividend_yield: snapshot.dividend_yield, roe: snapshot.roe };
  const topPeers = [...peers].sort((a, b) => (b.market_cap ?? 0) - (a.market_cap ?? 0)).slice(0, PEER_CHART_LIMIT);
  const combined = [selfRow, ...topPeers];

  const charts = PEER_RATIO_METRICS.map((m) => {
    // market_cap of exactly 0 means missing shares-outstanding data upstream,
    // not a real zero-valued company - exclude rather than plot a false bar.
    const data = combined
      .filter((row) => (m.key === "market_cap" ? row[m.key] > 0 : row[m.key] != null))
      .map((row) => ({ ticker: row.ticker, value: m.scale ? row[m.key] * m.scale : row[m.key] }));
    return { ...m, data };
  }).filter((c) => c.data.length > 1);

  if (charts.length === 0) return null;

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Peer Ratio Comparison</CardTitle>
        <p className="text-xs text-muted-foreground">{ticker} vs. its largest sector peers by market cap.</p>
      </CardHeader>
      <CardContent className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        {charts.map((c) => (
          <PeerComparisonChart key={c.key} ticker={ticker} label={c.label} formatter={c.formatter} data={c.data} />
        ))}
      </CardContent>
    </Card>
  );
}

function RatiosSection({ snapshot }) {
  if (!snapshot) return <Skeleton className="h-64 w-full" />;
  const hasSectorComparison = snapshot.sector_pe_avg != null || snapshot.sector_pb_avg != null;
  return (
    <div className="space-y-6">
      {hasSectorComparison && (
        <div>
          <h3 className="mb-3 text-sm font-semibold">This Company vs. Sector Average</h3>
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            {snapshot.sector_pe_avg != null && (
              <div>
                <p className="mb-1 text-xs text-muted-foreground">P/E Ratio</p>
                <SectorRatioChart label="P/E" companyValue={snapshot.pe_ratio} sectorValue={snapshot.sector_pe_avg} />
              </div>
            )}
            {snapshot.sector_pb_avg != null && (
              <div>
                <p className="mb-1 text-xs text-muted-foreground">P/B Ratio</p>
                <SectorRatioChart label="P/B" companyValue={snapshot.pb_ratio} sectorValue={snapshot.sector_pb_avg} />
              </div>
            )}
          </div>
        </div>
      )}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <SnapshotStat label="ROE" value={snapshot.roe != null ? `${(snapshot.roe * 100).toFixed(2)}%` : "—"} />
        <SnapshotStat label="Dividend Yield" value={snapshot.dividend_yield != null ? `${(snapshot.dividend_yield * 100).toFixed(2)}%` : "—"} />
        <SnapshotStat label="EPS (TTM)" value={snapshot.eps_trailing != null ? `Rs. ${snapshot.eps_trailing.toFixed(2)}` : "—"} />
        <SnapshotStat label="P/B Ratio" value={snapshot.pb_ratio != null ? snapshot.pb_ratio.toFixed(2) : "—"} />
      </div>
    </div>
  );
}

function PerformanceChart({ stockReturns, benchmarkReturns, benchmarkName }) {
  const chartData = PERFORMANCE_WINDOW_KEYS.map((w) => ({
    window: w,
    "This Stock": stockReturns[w],
    [benchmarkName]: benchmarkReturns[w],
  }));
  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 0, left: 10, bottom: 0 }}>
          <CartesianGrid stroke="var(--border)" strokeOpacity={0.6} vertical={false} strokeDasharray="4 4" />
          <XAxis dataKey="window" stroke="var(--muted-foreground)" tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} />
          <YAxis stroke="var(--muted-foreground)" tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} tickFormatter={(v) => `${v}%`} />
          <Tooltip
            cursor={{ fill: "var(--muted)" }}
            contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", borderRadius: "8px", color: "var(--foreground)" }}
            formatter={(v) => (v != null ? `${v.toFixed(2)}%` : "—")}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="This Stock" fill="var(--primary)" radius={[3, 3, 0, 0]} />
          <Bar dataKey={benchmarkName} fill="var(--muted-foreground)" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function FinancialsChart({ fundamentals }) {
  const chartData = [...fundamentals].reverse().slice(0, 12).reverse().map((r) => ({
    period: r.report_date,
    Revenue: r.revenue,
    "Net Income": r.net_income,
  }));
  if (chartData.length === 0) return null;
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} margin={{ top: 10, right: 0, left: 10, bottom: 0 }}>
          <CartesianGrid stroke="var(--border)" strokeOpacity={0.6} vertical={false} strokeDasharray="4 4" />
          <XAxis dataKey="period" stroke="var(--muted-foreground)" tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} />
          <YAxis stroke="var(--muted-foreground)" tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} tickFormatter={fmtCompact} />
          <Tooltip
            contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", borderRadius: "8px", color: "var(--foreground)" }}
            formatter={(v) => fmtCompact(v)}
          />
          <Bar dataKey="Revenue" fill="var(--primary)" radius={[3, 3, 0, 0]} />
          <Bar dataKey="Net Income" fill="var(--positive)" radius={[3, 3, 0, 0]} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function DividendsSection({ dividends }) {
  const { sortedRows, sortKey, direction, requestSort } = useSort(dividends, "ex_date", "desc");
  const { pageItems, page, totalPages, hasPrev, hasNext, goPrev, goNext } = usePagination(sortedRows, 5);

  if (!dividends || dividends.length === 0) {
    return <p className="text-sm text-muted-foreground">No dividend history available.</p>;
  }
  const chartData = [...dividends]
    .reverse()
    .filter((d) => d.amount != null)
    .map((d) => ({ date: d.ex_date, Amount: d.amount }));

  const sortHeadProps = { activeKey: sortKey, direction, onSort: requestSort };

  return (
    <div className="space-y-4">
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 0, left: 10, bottom: 0 }}>
            <CartesianGrid stroke="var(--border)" strokeOpacity={0.6} vertical={false} strokeDasharray="4 4" />
            <XAxis dataKey="date" stroke="var(--muted-foreground)" tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} minTickGap={30} />
            <YAxis stroke="var(--muted-foreground)" tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} />
            <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", borderRadius: "8px", color: "var(--foreground)" }} />
            <Bar dataKey="Amount" fill="var(--primary)" radius={[3, 3, 0, 0]} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <SortableTableHead label="Ex-Date" sortKey="ex_date" {...sortHeadProps} />
            <SortableTableHead label="Type" sortKey="type" {...sortHeadProps} />
            <SortableTableHead label="Amount (Rs.)" sortKey="amount" align="right" {...sortHeadProps} />
          </TableRow>
        </TableHeader>
        <TableBody>
          {pageItems.map((d, idx) => (
            <TableRow key={idx}>
              <TableCell>{d.ex_date || "—"}</TableCell>
              <TableCell className="text-muted-foreground">{d.type || "—"}</TableCell>
              <TableCell className="text-right tabular-nums">{d.amount != null ? d.amount.toFixed(2) : "—"}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <PaginationControls page={page} totalPages={totalPages} hasPrev={hasPrev} hasNext={hasNext} onPrev={goPrev} onNext={goNext} />
    </div>
  );
}

const FUNDAMENTALS_COLUMNS = [
  { key: "revenue", label: "Revenue" },
  { key: "net_income", label: "Net Income" },
  { key: "eps", label: "EPS" },
  { key: "roe", label: "ROE" },
  { key: "pe_ratio", label: "P/E" },
];

function FundamentalsSection({ fundamentals }) {
  const rows = [...fundamentals].reverse().slice(0, 8);
  const { sortedRows, sortKey, direction, requestSort } = useSort(rows, "report_date", "desc");
  if (rows.length === 0) return <p className="text-sm text-muted-foreground">No fundamentals data available yet.</p>;
  const sortHeadProps = { activeKey: sortKey, direction, onSort: requestSort };
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <SortableTableHead label="Report Date" sortKey="report_date" {...sortHeadProps} />
          {FUNDAMENTALS_COLUMNS.map((c) => (
            <SortableTableHead key={c.key} label={c.label} sortKey={c.key} align="right" {...sortHeadProps} />
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {sortedRows.map((row) => (
          <TableRow key={row.report_date}>
            <TableCell className="font-medium">{row.report_date}</TableCell>
            {FUNDAMENTALS_COLUMNS.map((c) => (
              <TableCell key={c.key} className="text-right tabular-nums">
                {typeof row[c.key] === "number" ? row[c.key].toLocaleString(undefined, { maximumFractionDigits: 2 }) : "—"}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function EventsSection({ events }) {
  const { pageItems, page, totalPages, hasPrev, hasNext, goPrev, goNext } = usePagination(events, 5);
  if (!events || events.length === 0) return <p className="text-sm text-muted-foreground">No recent corporate events.</p>;
  return (
    <div className="space-y-2">
      {pageItems.map((e, idx) => (
        <div key={idx} className="flex items-start justify-between gap-3 border-b border-border pb-2 text-sm last:border-0">
          <div>
            <div className="font-medium">{e.title}</div>
            <div className="text-xs text-muted-foreground">{e.date}</div>
          </div>
          <Badge variant="outline">{e.type}</Badge>
        </div>
      ))}
      <PaginationControls page={page} totalPages={totalPages} hasPrev={hasPrev} hasNext={hasNext} onPrev={goPrev} onNext={goNext} />
    </div>
  );
}

function ReportsSection({ reports }) {
  const { pageItems, page, totalPages, hasPrev, hasNext, goPrev, goNext } = usePagination(reports, 5);
  if (reports === null) return <Skeleton className="h-64 w-full" />;
  if (reports.length === 0) return <p className="text-sm text-muted-foreground">No filings found on PSX's Financial Portal for the last few years.</p>;
  return (
    <div className="space-y-2">
      {pageItems.map((r, idx) => (
        <a
          key={idx}
          href={r.download_url}
          target="_blank"
          rel="noreferrer"
          className="flex items-center justify-between gap-3 rounded-lg border border-border p-3 text-sm transition-colors hover:bg-muted"
        >
          <div className="flex items-center gap-3">
            <FileDown className="h-4 w-4 shrink-0 text-primary" />
            <div>
              <div className="font-medium">{r.type} Report — {r.period_ended}</div>
              <div className="text-xs text-muted-foreground">Filed {r.posting_date}</div>
            </div>
          </div>
          <Badge variant="outline">PDF</Badge>
        </a>
      ))}
      <PaginationControls page={page} totalPages={totalPages} hasPrev={hasPrev} hasNext={hasNext} onPrev={goPrev} onNext={goNext} />
    </div>
  );
}

export function CompanyDetail() {
  const { ticker } = useParams();
  const [data, setData] = useState(null);
  const [company, setCompany] = useState(null);
  const [liveData, setLiveData] = useState(null);
  const [fundamentals, setFundamentals] = useState([]);
  const [events, setEvents] = useState([]);
  const [snapshot, setSnapshot] = useState(null);
  const [dividends, setDividends] = useState([]);
  const [indices, setIndices] = useState([]);
  const [peers, setPeers] = useState(null);
  const [reports, setReports] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [modelType, setModelType] = useState("RF");
  const [timeRange, setTimeRange] = useState("30D");
  const [activeSection, setActiveSection] = useState("overview");
  const marketClosed = useMemo(isMarketClosedNow, []);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      axios.post(`${API_BASE_URL}/api/predict`, { ticker }),
      axios.get(`${API_BASE_URL}/api/company/${ticker}`),
      axios.get(`${API_BASE_URL}/api/realtime/${ticker}`).catch(() => null),
      axios.get(`${API_BASE_URL}/api/company/${ticker}/fundamentals`).catch(() => null),
      axios.get(`${API_BASE_URL}/api/company/${ticker}/events`).catch(() => null),
      axios.get(`${API_BASE_URL}/api/company/${ticker}/snapshot`).catch(() => null),
      axios.get(`${API_BASE_URL}/api/company/${ticker}/dividends`).catch(() => null),
      axios.get(`${API_BASE_URL}/api/indices`).catch(() => null),
    ])
      .then(([predRes, compRes, liveRes, fundRes, eventsRes, snapRes, divRes, indicesRes]) => {
        setData(predRes.data);
        setCompany(compRes.data);
        setLiveData(liveRes?.data ?? null);
        setFundamentals(fundRes?.data?.fundamentals ?? []);
        setEvents(eventsRes?.data?.events ?? []);
        setSnapshot(snapRes?.data ?? null);
        setDividends(divRes?.data?.dividends ?? []);
        setIndices(indicesRes?.data?.indices ?? []);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed to load this company."))
      .finally(() => setLoading(false));
  }, [ticker]);

  useEffect(() => {
    if (!company?.sector) return;
    setPeers(null);
    axios
      .get(`${API_BASE_URL}/api/sectors/${encodeURIComponent(company.sector)}`)
      .then((res) => setPeers((res.data.tickers ?? []).filter((t) => t.ticker !== ticker)))
      .catch((err) => console.error("Failed to fetch sector peers", err));
  }, [company?.sector, ticker]);

  useEffect(() => {
    setReports(null);
    axios
      .get(`${API_BASE_URL}/api/company/${ticker}/reports`)
      .then((res) => setReports(res.data.reports ?? []))
      .catch((err) => {
        console.error("Failed to fetch company reports", err);
        setReports([]);
      });
  }, [ticker]);

  useEffect(() => {
    if (!data) return;
    const elements = SECTIONS.map((s) => document.getElementById(s.id)).filter(Boolean);
    if (elements.length === 0) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length > 0) {
          setActiveSection(visible[0].target.id);
        }
      },
      { rootMargin: "-100px 0px -70% 0px", threshold: 0 }
    );
    elements.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [data, ticker]);

  const benchmark = indices.find((i) => i.key === "all_share") ?? null;
  const stockReturns = useMemo(
    () => computeReturns((data?.historical_data ?? []).map((h) => ({ date: h.date, value: h.close }))),
    [data]
  );
  const benchmarkReturns = useMemo(
    () => computeReturns((benchmark?.history ?? []).map((h) => ({ date: h.date, value: h.level }))),
    [benchmark]
  );

  useEffect(() => {
    if (!data) return;
    const interval = setInterval(() => {
      axios
        .get(`${API_BASE_URL}/api/realtime/${ticker}`)
        .then((res) => setLiveData(res.data))
        .catch((err) => console.error("Failed to refresh live price", err));
    }, 15000);
    return () => clearInterval(interval);
  }, [data, ticker]);

  if (loading) return <Skeleton className="h-96 w-full" />;
  if (error) return <p className="rounded-lg border border-negative/30 bg-negative/10 p-4 text-negative">{error}</p>;
  if (!data || !company) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <CompanyLogo ticker={ticker} size="lg" />
        <div>
          <h1 className="text-2xl font-bold leading-tight">{company.name}</h1>
          <div className="mt-1 flex items-center gap-2 text-sm text-muted-foreground">
            <Badge variant="outline">{ticker}</Badge>
            <span>{company.sector}</span>
          </div>
        </div>
      </div>

      <SectionNav activeId={activeSection} />

      <div className="space-y-10">
        <section id="overview" className="scroll-mt-16 space-y-4">
          <h2 className="text-lg font-bold">Overview</h2>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="space-y-4 md:col-span-1">
              <PriceCard data={data} liveData={liveData} marketClosed={marketClosed} />
              <PredictionCard data={data} modelType={modelType} setModelType={setModelType} />
              <SentimentCard sentiment={data.latest_sentiment} />
              <DividendCard dividend={data.latest_dividend} currentPrice={data.current_price} />
            </div>
            <div className="md:col-span-2">
              <PriceChart data={data} modelType={modelType} timeRange={timeRange} setTimeRange={setTimeRange} />
            </div>
          </div>

          <CompanySnapshot snapshot={snapshot} />

          <FiftyTwoWeekRangeBar low={snapshot?.week_52_low} high={snapshot?.week_52_high} current={data.current_price} />

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <PerformanceRow stockReturns={stockReturns} benchmarkReturns={benchmarkReturns} benchmarkName={benchmark?.name ?? "All Share Index"} />
            </div>
            <InsiderActivityCard snapshot={snapshot} />
          </div>

          {data.recent_news && data.recent_news.length > 0 && (
            <Card className="shadow-sm">
              <CardHeader><CardTitle>Latest News</CardTitle></CardHeader>
              <CardContent className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
                {data.recent_news.map((news, idx) => (
                  <a key={idx} href={news.url} target="_blank" rel="noreferrer" className="flex flex-col justify-between rounded-lg border border-border p-3 transition-colors hover:bg-muted">
                    <div>
                      <div className="mb-1 flex items-center justify-between">
                        <span className="text-[10px] font-bold uppercase text-muted-foreground">{news.source}</span>
                        <Badge variant={news.sentiment_score > 0.05 ? "positive" : news.sentiment_score < -0.05 ? "negative" : "muted"}>
                          {news.sentiment_score > 0.05 ? "Bullish" : news.sentiment_score < -0.05 ? "Bearish" : "Neutral"}
                        </Badge>
                      </div>
                      <h4 className="line-clamp-2 text-sm font-semibold">{news.headline}</h4>
                    </div>
                    <div className="mt-3 text-[10px] text-muted-foreground">
                      {new Date(news.published_at).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                    </div>
                  </a>
                ))}
              </CardContent>
            </Card>
          )}
        </section>

        <section id="financials" className="scroll-mt-16 space-y-4">
          <h2 className="text-lg font-bold">Financials</h2>
          <Card className="shadow-sm">
            <CardHeader><CardTitle>Revenue & Net Income Trend</CardTitle></CardHeader>
            <CardContent><FinancialsChart fundamentals={fundamentals} /></CardContent>
          </Card>
          <Card className="shadow-sm">
            <CardHeader><CardTitle>Fundamentals History</CardTitle></CardHeader>
            <CardContent><FundamentalsSection fundamentals={fundamentals} /></CardContent>
          </Card>
        </section>

        <section id="ratios" className="scroll-mt-16 space-y-4">
          <h2 className="text-lg font-bold">Ratios</h2>
          <Card className="shadow-sm">
            <CardHeader><CardTitle>Valuation Ratios</CardTitle></CardHeader>
            <CardContent><RatiosSection snapshot={snapshot} /></CardContent>
          </Card>
        </section>

        <section id="dividends" className="scroll-mt-16 space-y-4">
          <h2 className="text-lg font-bold">Dividends</h2>
          <Card className="shadow-sm">
            <CardHeader><CardTitle>Dividend History</CardTitle></CardHeader>
            <CardContent><DividendsSection dividends={dividends} /></CardContent>
          </Card>
        </section>

        <section id="peers" className="scroll-mt-16 space-y-4">
          <h2 className="text-lg font-bold">Peers</h2>
          <PeerRatioComparison ticker={ticker} snapshot={snapshot} peers={peers} />
          <Card className="shadow-sm">
            <CardHeader><CardTitle>Sector Peers — {company.sector}</CardTitle></CardHeader>
            <CardContent>
              {peers === null ? (
                <Skeleton className="h-64 w-full" />
              ) : peers.length === 0 ? (
                <p className="text-sm text-muted-foreground">No other tracked companies in this sector.</p>
              ) : (
                <ScreenerTable rows={peers} />
              )}
            </CardContent>
          </Card>
        </section>

        <section id="events" className="scroll-mt-16 space-y-4">
          <h2 className="text-lg font-bold">Events</h2>
          <Card className="shadow-sm">
            <CardHeader><CardTitle>Corporate Events</CardTitle></CardHeader>
            <CardContent><EventsSection events={events} /></CardContent>
          </Card>
        </section>

        <section id="reports" className="scroll-mt-16 space-y-4">
          <h2 className="text-lg font-bold">Reports</h2>
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle>Financial Reports</CardTitle>
              <p className="text-xs text-muted-foreground">Official filings linked directly from PSX's Financial Portal.</p>
            </CardHeader>
            <CardContent><ReportsSection reports={reports} /></CardContent>
          </Card>
        </section>

        <section id="profile" className="scroll-mt-16 space-y-4">
          <h2 className="text-lg font-bold">Profile</h2>
          <Card className="shadow-sm">
            <CardHeader><CardTitle>Company Profile</CardTitle></CardHeader>
            <CardContent className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="space-y-4">
                <div>
                  <h3 className="mb-1 text-xs font-semibold uppercase text-muted-foreground">Description</h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">{company.description || "No description available."}</p>
                </div>
                {company.details?.ADDRESS && (
                  <div>
                    <h3 className="mb-1 text-xs font-semibold uppercase text-muted-foreground">Address</h3>
                    <p className="text-sm">{company.details.ADDRESS}</p>
                  </div>
                )}
                {company.details?.WEBSITE && (
                  <div>
                    <h3 className="mb-1 text-xs font-semibold uppercase text-muted-foreground">Website</h3>
                    <a href={company.details.WEBSITE.startsWith("http") ? company.details.WEBSITE : `http://${company.details.WEBSITE}`} target="_blank" rel="noreferrer" className="text-sm font-medium text-primary hover:underline">
                      {company.details.WEBSITE}
                    </a>
                  </div>
                )}
              </div>
              <div className="space-y-4">
                {company.people && company.people.length > 0 && (
                  <div>
                    <h3 className="mb-1 text-xs font-semibold uppercase text-muted-foreground">Key People</h3>
                    <div className="rounded-lg border border-border">
                      {company.people.map((p, idx) => (
                        <div key={idx} className="flex justify-between border-b border-border px-3 py-2 text-sm last:border-0">
                          <span className="font-medium">{p.name}</span>
                          <span className="text-muted-foreground">{p.role}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {company.details?.AUDITOR && (
                  <div>
                    <h3 className="mb-1 text-xs font-semibold uppercase text-muted-foreground">Auditor</h3>
                    <p className="text-sm">{company.details.AUDITOR}</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}

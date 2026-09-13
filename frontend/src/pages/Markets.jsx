import React, { useEffect, useMemo, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { API_BASE_URL } from "../config";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Skeleton } from "../components/ui/skeleton";
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "../components/ui/table";
import { PaginationControls } from "../components/ui/pagination";
import { RANGE_OPTIONS, filterHistoryByRange, formatChartDate } from "../lib/dateRange";
import { CHART_COLORS } from "../lib/chartColors";

export const MARKET_TABS = [
  { key: "indices", label: "Indices" },
  { key: "forex", label: "Forex" },
  { key: "commodities", label: "Commodities" },
  { key: "economy", label: "Economy" },
  { key: "announcements", label: "Announcements" },
];

// Groups items (already fetched, flat) by similarity so related series land
// on one comparable chart instead of a wall of disconnected mini-sparklines.
// A group simply doesn't render if every one of its keys got excluded
// upstream for being stale/absent - no need to hand-track which fields exist.
const COMMODITY_GROUPS = [
  { title: "Energy", keys: ["brent_oil_price", "wti_oil_price", "gas_price", "lng_price", "coal_price"] },
  { title: "Metals", keys: ["gold_price", "copper_price", "aluminum_price", "steel_price"] },
  { title: "Agriculture", keys: ["cotton_price", "wheat_price", "urea_price", "palm_oil_price"] },
];

const ECONOMY_GROUPS = [
  { title: "Inflation (CPI)", keys: ["cpi_headline", "cpi_core", "cpi_food"] },
  { title: "Interest Rates", keys: ["sbp_policy_rate", "kibor_3m", "kibor_6m", "kibor_1y"] },
  { title: "Reserves & Money Supply", keys: ["sbp_reserves", "total_fx_reserves", "m2_money_supply"] },
  { title: "Remittances", keys: ["monthly_remittances", "remittances_saudi", "remittances_uae", "remittances_usa", "remittances_uk"] },
];

const FOREX_GROUPS = [
  { title: "Exchange Rates (to PKR)", keys: ["pkr_usd_rate", "eur_pkr_rate", "gbp_pkr_rate", "cny_pkr_rate"] },
];

function groupItems(items, groups) {
  const byKey = Object.fromEntries((items ?? []).map((i) => [i.key, i]));
  return groups
    .map((g) => ({ title: g.title, items: g.keys.map((k) => byKey[k]).filter(Boolean) }))
    .filter((g) => g.items.length > 0);
}

function RangeSelector({ range, setRange }) {
  return (
    <div className="flex gap-1 rounded-lg bg-muted p-1">
      {RANGE_OPTIONS.map((r) => (
        <button
          key={r}
          onClick={() => setRange(r)}
          className={`rounded-md px-2.5 py-1 text-xs font-semibold transition-colors ${range === r ? "bg-card text-foreground shadow-sm" : "text-muted-foreground"}`}
        >
          {r}
        </button>
      ))}
    </div>
  );
}

function fmtValue(value) {
  return typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "—";
}

// A single item's own actual-value chart, own auto-scaled axis. Items in a
// group are related by category (e.g. "Metals") but not necessarily by unit
// (USD/oz vs USD/lb vs USD/ton) or by scale (Gold ~4400 vs Copper ~6.6) - a
// combined chart plotting real values would flatten the smaller series to an
// invisible line near zero, so each gets its own chart instead.
function ItemChart({ item, range, color }) {
  const history = useMemo(() => {
    const trimmed = filterHistoryByRange(item.history, range);
    // Monthly/quarterly-cadence data (remittances, SBP reserves) can have
    // 0-1 points inside a short 7D/30D window - fall back to the full
    // series rather than rendering a blank chart.
    return trimmed.length >= 2 ? trimmed : (item.history ?? []);
  }, [item.history, range]);

  if (history.length < 2) return null;

  return (
    <div>
      <p className="mb-1 flex items-baseline justify-between text-xs">
        <span className="font-medium text-foreground">{item.label}</span>
        <span className="text-muted-foreground">{item.unit}</span>
      </p>
      <div className="h-40 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--border)" strokeOpacity={0.5} vertical={false} strokeDasharray="4 4" />
            <XAxis
              dataKey="date"
              stroke="var(--muted-foreground)"
              tick={{ fill: "var(--muted-foreground)", fontSize: 10 }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              tickFormatter={(v) => formatChartDate(v, range)}
              minTickGap={30}
            />
            <YAxis
              domain={["auto", "auto"]}
              stroke="var(--muted-foreground)"
              tick={{ fill: "var(--muted-foreground)", fontSize: 10 }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              width={48}
              tickFormatter={(v) => fmtValue(v)}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", borderRadius: "8px", color: "var(--foreground)", fontSize: 12 }}
              labelFormatter={(label) => label}
              formatter={(v) => [`${fmtValue(v)}${item.unit ? ` ${item.unit}` : ""}`, item.label]}
            />
            <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function GroupSection({ title, items, range }) {
  const hasChange = items.some((item) => item.changePercent != null);

  return (
    <Card className="shadow-sm">
      <CardHeader><CardTitle>{title}</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {items.map((item, idx) => (
            <ItemChart key={item.key} item={item} range={range} color={CHART_COLORS[idx % CHART_COLORS.length]} />
          ))}
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead className="text-right">Value</TableHead>
              <TableHead>Unit</TableHead>
              {hasChange && <TableHead className="text-right">Change</TableHead>}
              <TableHead className="text-right">As Of</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((item) => (
              <TableRow key={item.key}>
                <TableCell className="font-medium">{item.label}</TableCell>
                <TableCell className="text-right tabular-nums">{fmtValue(item.value)}</TableCell>
                <TableCell className="text-muted-foreground">{item.unit || "—"}</TableCell>
                {hasChange && (
                  <TableCell className={`text-right tabular-nums ${item.changePercent == null ? "text-muted-foreground" : item.changePercent >= 0 ? "text-positive" : "text-negative"}`}>
                    {item.changePercent != null ? `${item.changePercent >= 0 ? "+" : ""}${item.changePercent.toFixed(2)}%` : "—"}
                  </TableCell>
                )}
                <TableCell className="text-right text-muted-foreground">{item.as_of}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

function GroupedTab({ items, groups, range, setRange }) {
  if (items === null) {
    return (
      <div className="space-y-3">
        <div className="flex justify-end"><Skeleton className="h-8 w-48" /></div>
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }
  const grouped = groupItems(items, groups);
  if (grouped.length === 0) {
    return <p className="py-8 text-center text-sm text-muted-foreground">No current data available for this section.</p>;
  }
  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <RangeSelector range={range} setRange={setRange} />
      </div>
      {grouped.map((g) => (
        <GroupSection key={g.title} title={g.title} items={g.items} range={range} />
      ))}
    </div>
  );
}

function IndicesTab() {
  const [indices, setIndices] = useState(null);
  const [range, setRange] = useState("30D");

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/api/indices`)
      .then((res) => setIndices((res.data.indices ?? []).filter((i) => i.level != null)))
      .catch(() => setIndices([]));
  }, []);

  const items = indices?.map((idx) => ({
    key: idx.key,
    label: idx.name,
    value: idx.level,
    unit: null,
    changePercent: idx.change_percent,
    as_of: idx.history?.[idx.history.length - 1]?.date ?? "—",
    history: (idx.history ?? []).map((h) => ({ date: h.date, value: h.level })),
  })) ?? null;

  return <GroupedTab items={items} groups={[{ title: "PSX Indices", keys: items?.map((i) => i.key) ?? [] }]} range={range} setRange={setRange} />;
}

function useMacroGroup(endpoint) {
  const [items, setItems] = useState(null);
  useEffect(() => {
    setItems(null);
    axios
      .get(`${API_BASE_URL}${endpoint}`)
      .then((res) => setItems(res.data.items ?? []))
      .catch(() => setItems([]));
  }, [endpoint]);
  return items;
}

function ForexTab() {
  const [range, setRange] = useState("30D");
  return <GroupedTab items={useMacroGroup("/api/forex")} groups={FOREX_GROUPS} range={range} setRange={setRange} />;
}

function CommoditiesTab() {
  const [range, setRange] = useState("30D");
  return <GroupedTab items={useMacroGroup("/api/commodities")} groups={COMMODITY_GROUPS} range={range} setRange={setRange} />;
}

function EconomyTab() {
  const [range, setRange] = useState("1Y");
  return <GroupedTab items={useMacroGroup("/api/economy")} groups={ECONOMY_GROUPS} range={range} setRange={setRange} />;
}

const ANNOUNCEMENTS_PAGE_SIZE = 20;

function AnnouncementsTab() {
  const [page, setPage] = useState(1);
  const [data, setData] = useState(null);

  useEffect(() => {
    setData(null);
    axios
      .get(`${API_BASE_URL}/api/announcements`, {
        params: { limit: ANNOUNCEMENTS_PAGE_SIZE, offset: (page - 1) * ANNOUNCEMENTS_PAGE_SIZE },
      })
      .then((res) => setData(res.data.announcements ?? []))
      .catch(() => setData([]));
  }, [page]);

  if (data === null) return <Skeleton className="h-96 w-full" />;
  if (data.length === 0 && page === 1) {
    return <p className="py-8 text-center text-sm text-muted-foreground">No announcements available.</p>;
  }

  return (
    <div className="space-y-2">
      {data.map((a, idx) => (
        <div key={idx} className="flex items-start justify-between gap-3 border-b border-border pb-2 text-sm last:border-0">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Link to={`/company/${a.ticker}`} className="font-semibold text-primary hover:underline">{a.ticker}</Link>
              <Badge variant="outline">{a.type}</Badge>
            </div>
            <div className="mt-0.5 truncate text-sm">{a.title}</div>
            <div className="text-xs text-muted-foreground">{a.date}</div>
          </div>
        </div>
      ))}
      <PaginationControls
        page={page}
        // The API is offset/limit based with no total count, so this is an
        // estimate: assume one more page exists whenever the current page
        // came back full. hasPrev/hasNext (not this number) drive the
        // actual button states.
        totalPages={data.length === ANNOUNCEMENTS_PAGE_SIZE ? page + 1 : page}
        hasPrev={page > 1}
        hasNext={data.length === ANNOUNCEMENTS_PAGE_SIZE}
        onPrev={() => setPage((p) => Math.max(1, p - 1))}
        onNext={() => setPage((p) => p + 1)}
      />
    </div>
  );
}

const TAB_CONTENT = {
  indices: IndicesTab,
  forex: ForexTab,
  commodities: CommoditiesTab,
  economy: EconomyTab,
  announcements: AnnouncementsTab,
};

export function Markets() {
  const { tab } = useParams();

  if (!tab || !TAB_CONTENT[tab]) {
    return <Navigate to="/markets/indices" replace />;
  }

  const TabContent = TAB_CONTENT[tab];

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold">Markets</h1>
        <p className="text-sm text-muted-foreground">Real indices, forex, commodities, and economic indicators alongside PSX equities.</p>
      </div>

      <div className="flex gap-1 overflow-x-auto rounded-lg bg-muted p-1">
        {MARKET_TABS.map((t) => (
          <Link
            key={t.key}
            to={`/markets/${t.key}`}
            className={`shrink-0 whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              tab === t.key ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {t.label}
          </Link>
        ))}
      </div>

      {tab === "announcements" ? (
        <Card className="shadow-sm">
          <CardContent className="pt-5">
            <TabContent />
          </CardContent>
        </Card>
      ) : (
        <TabContent />
      )}
    </div>
  );
}

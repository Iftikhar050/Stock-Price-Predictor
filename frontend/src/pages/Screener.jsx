import React, { useEffect, useState } from "react";
import axios from "axios";
import { Sparkles, List, TrendingUp, TrendingDown, Activity } from "lucide-react";
import { API_BASE_URL } from "../config";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import { Skeleton } from "../components/ui/skeleton";
import { ScreenerTable } from "../components/ScreenerTable";
import { cn } from "../lib/utils";

const PREDICT_CHUNK_SIZE = 5;

// "52 Week High/Low Stocks" ranks by how close price sits to its own 52-week
// high/low (pct_from_52w_high/low, computed server-side) - not a separate
// filter, just a sort_by/order combination the backend already supports.
const PRESETS = [
  { key: "all", label: "All Stocks", sort_by: "market_cap", order: "desc", icon: List },
  { key: "top_gainers", label: "Top Gainers", sort_by: "change_percent", order: "desc", icon: TrendingUp },
  { key: "top_losers", label: "Top Losers", sort_by: "change_percent", order: "asc", icon: TrendingDown },
  { key: "most_active", label: "Most Active", sort_by: "volume", order: "desc", icon: Activity },
  { key: "52w_high", label: "52 Week High Stocks", sort_by: "pct_from_52w_high", order: "desc" },
  { key: "52w_low", label: "52 Week Low Stocks", sort_by: "pct_from_52w_low", order: "asc" },
  { key: "large_cap", label: "Large Cap Stocks", sort_by: "market_cap", order: "desc" },
  { key: "div_yield", label: "Top Dividend Yield", sort_by: "dividend_yield", order: "desc" },
  { key: "pb", label: "Top P/B Stocks", sort_by: "pb_ratio", order: "desc" },
  { key: "pe", label: "Top P/E Stocks", sort_by: "pe_ratio", order: "desc" },
];

const QUICK_TAB_KEYS = ["all", "top_gainers", "top_losers", "most_active"];

function QuickTabBar({ preset, onSelect }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {PRESETS.filter((p) => QUICK_TAB_KEYS.includes(p.key)).map((p) => {
        const Icon = p.icon;
        const active = preset === p.key;
        return (
          <button
            key={p.key}
            onClick={() => onSelect(p.key)}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full px-4 py-2 text-sm font-medium transition-colors",
              active ? "bg-foreground text-background" : "text-muted-foreground hover:bg-muted"
            )}
          >
            <Icon className="h-3.5 w-3.5" /> {p.label}
          </button>
        );
      })}
    </div>
  );
}

function SidebarSection({ title, children }) {
  return (
    <div className="border-b border-border py-4 first:pt-0 last:border-0">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">{title}</h3>
      {children}
    </div>
  );
}

function RadioRow({ active, onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors",
        active ? "bg-accent font-medium text-accent-foreground" : "text-muted-foreground hover:bg-muted"
      )}
    >
      <span
        className={cn(
          "h-3 w-3 shrink-0 rounded-full border-2",
          active ? "border-primary bg-primary" : "border-input"
        )}
      />
      {children}
    </button>
  );
}

export function Screener() {
  const [sectors, setSectors] = useState([]);
  const [sector, setSector] = useState("");
  const [minPe, setMinPe] = useState("");
  const [maxPe, setMaxPe] = useState("");
  const [preset, setPreset] = useState("all");
  const [page, setPage] = useState(1);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predictions, setPredictions] = useState(null);
  const [predictionsLoading, setPredictionsLoading] = useState(false);
  const pageSize = 25;

  const activePreset = PRESETS.find((p) => p.key === preset) ?? PRESETS[0];

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/api/sectors`)
      .then((res) => setSectors(res.data.sectors.map((s) => s.sector)))
      .catch((err) => console.error("Failed to fetch sectors", err));
  }, []);

  useEffect(() => {
    setLoading(true);
    const params = { sort_by: activePreset.sort_by, order: activePreset.order, page, page_size: pageSize };
    if (sector) params.sector = sector;
    if (minPe) params.min_pe = minPe;
    if (maxPe) params.max_pe = maxPe;

    axios
      .get(`${API_BASE_URL}/api/screener`, { params })
      .then((res) => {
        setResult(res.data);
        setPredictions(null); // rows changed - stale predictions no longer line up
      })
      .catch((err) => console.error("Failed to fetch screener", err))
      .finally(() => setLoading(false));
  }, [sector, minPe, maxPe, preset, page]);

  const totalPages = result ? Math.max(1, Math.ceil(result.total / pageSize)) : 1;

  // Opt-in only: the old dashboard auto-ran /api/predict for all 99 tickers on
  // every load and every 60s poll. This runs once, on click, only for the up-to
  // 25 tickers currently visible on this page - a bounded, deliberate action.
  const runPredictions = async () => {
    const tickers = (result?.results ?? []).map((r) => r.ticker);
    if (tickers.length === 0) return;
    setPredictionsLoading(true);
    const collected = {};
    for (let i = 0; i < tickers.length; i += PREDICT_CHUNK_SIZE) {
      const chunk = tickers.slice(i, i + PREDICT_CHUNK_SIZE);
      const responses = await Promise.all(
        chunk.map((t) => axios.post(`${API_BASE_URL}/api/predict`, { ticker: t }).catch(() => null))
      );
      responses.forEach((res, idx) => {
        if (res?.data) collected[chunk[idx]] = res.data;
      });
      setPredictions({ ...collected });
    }
    setPredictionsLoading(false);
  };

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold">Screener</h1>
        <p className="text-sm text-muted-foreground">Filter and rank all KSE-100 companies by fundamentals and price action.</p>
      </div>

      <QuickTabBar preset={preset} onSelect={(key) => { setPreset(key); setPage(1); }} />

      {result && (
        <p className="text-sm text-muted-foreground">
          <span className="font-semibold text-foreground">{result.total_tracked}</span> companies ·{" "}
          <span className="font-medium text-positive">{result.total_up} up</span> /{" "}
          <span className="font-medium text-negative">{result.total_down} down</span>
        </p>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-[240px_1fr]">
        <Card className="h-fit shadow-sm">
          <CardContent className="pt-5">
            <SidebarSection title="Valuation">
              {PRESETS.map((p) => (
                <RadioRow key={p.key} active={preset === p.key} onClick={() => { setPreset(p.key); setPage(1); }}>
                  {p.label}
                </RadioRow>
              ))}
            </SidebarSection>

            <SidebarSection title="Sector">
              <RadioRow active={sector === ""} onClick={() => { setSector(""); setPage(1); }}>
                All sectors
              </RadioRow>
              <div className="mt-1 max-h-64 space-y-0.5 overflow-y-auto">
                {sectors.map((s) => (
                  <RadioRow key={s} active={sector === s} onClick={() => { setSector(s); setPage(1); }}>
                    {s}
                  </RadioRow>
                ))}
              </div>
            </SidebarSection>

            <SidebarSection title="P/E Range">
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  placeholder="Min"
                  value={minPe}
                  onChange={(e) => { setMinPe(e.target.value); setPage(1); }}
                  className="h-8"
                />
                <span className="text-muted-foreground">–</span>
                <Input
                  type="number"
                  placeholder="Max"
                  value={maxPe}
                  onChange={(e) => { setMaxPe(e.target.value); setPage(1); }}
                  className="h-8"
                />
              </div>
            </SidebarSection>
          </CardContent>
        </Card>

        <Card className="min-w-0 shadow-sm">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>{result ? `${result.total} companies` : "Loading..."}</CardTitle>
            <Button
              variant="outline"
              size="sm"
              onClick={runPredictions}
              disabled={predictionsLoading || loading || !result?.results?.length}
              title="Experimental: the underlying model's directional accuracy has not been validated above chance. Not investment advice."
            >
              <Sparkles className="mr-1.5 h-3.5 w-3.5" />
              {predictionsLoading ? "Running predictions…" : "Run ML Predictions (this page)"}
            </Button>
          </CardHeader>
          <CardContent>
            {loading ? (
              <Skeleton className="h-64 w-full" />
            ) : (
              <>
                <ScreenerTable rows={result?.results ?? []} predictions={predictions} predictionsLoading={predictionsLoading} />
                <div className="mt-4 flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Page {page} of {totalPages}</span>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>Previous</Button>
                    <Button variant="outline" size="sm" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>Next</Button>
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

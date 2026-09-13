import React, { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import Select from "react-select";
import { Search } from "lucide-react";
import { API_BASE_URL } from "../config";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Skeleton } from "../components/ui/skeleton";
import { MoversTable } from "../components/MoversTable";
import { ScreenerTable } from "../components/ScreenerTable";
import { CuratedLists } from "../components/CuratedLists";
import { selectStyles } from "../lib/reactSelectStyles";

function IndexCard({ label, value, changePercent }) {
  return (
    <Card className="transition-shadow hover:shadow-md">
      <CardContent className="pt-5">
        <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{label}</div>
        <div className="mt-1 text-2xl font-bold tabular-nums">
          {value != null ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "—"}
        </div>
        {changePercent != null && (
          <div className={`mt-0.5 text-xs font-semibold tabular-nums ${changePercent >= 0 ? "text-positive" : "text-negative"}`}>
            {changePercent >= 0 ? "▲" : "▼"} {Math.abs(changePercent).toFixed(2)}%
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Hero({ tickerCount }) {
  const [tickers, setTickers] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/api/tickers`)
      .then((res) => setTickers(res.data.map((t) => ({ value: t.ticker, label: `${t.ticker} — ${t.name}` }))))
      .catch((err) => console.error("Failed to fetch tickers", err));
  }, []);

  return (
    <div className="-mx-4 -mt-6 border-b border-border bg-gradient-to-b from-accent/40 to-background px-4 pb-10 pt-14 text-center sm:px-6">
      <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl">
        Data-driven insight for every <span className="text-primary">PSX</span> company
      </h1>
      <p className="mx-auto mt-3 max-w-xl text-sm text-muted-foreground sm:text-base">
        Real-time prices, fundamentals, and a 4-model machine learning price signal — all in one place.
      </p>

      <div className="mx-auto mt-6 max-w-md text-left">
        <Select
          options={tickers}
          placeholder="Search a company..."
          onChange={(opt) => opt && navigate(`/company/${opt.value}`)}
          styles={selectStyles}
          components={{
            DropdownIndicator: () => <Search className="mr-2 h-4 w-4 text-muted-foreground" />,
          }}
          isClearable
        />
      </div>

      <div className="mx-auto mt-8 flex max-w-md items-center justify-center gap-8 text-sm">
        <Link to="/screener" className="rounded-md transition-opacity hover:opacity-70">
          <div className="text-xl font-bold tabular-nums">{tickerCount != null ? tickerCount : "—"}</div>
          <div className="text-xs text-muted-foreground underline decoration-dotted underline-offset-2">Companies tracked</div>
        </Link>
        <div>
          <div className="text-xl font-bold">4</div>
          <div className="text-xs text-muted-foreground">ML models per stock</div>
        </div>
        <div>
          <div className="text-xl font-bold">Live</div>
          <div className="text-xs text-muted-foreground">Market data</div>
        </div>
      </div>
    </div>
  );
}

export function Home() {
  const [sectors, setSectors] = useState([]);
  const [indices, setIndices] = useState(null);
  const [screenerPreview, setScreenerPreview] = useState(null);
  const [tickerCount, setTickerCount] = useState(null);

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/api/sectors`)
      .then((res) => setSectors(res.data.sectors))
      .catch((err) => console.error("Failed to fetch sectors", err));

    axios
      .get(`${API_BASE_URL}/api/indices`)
      .then((res) => setIndices(res.data.indices))
      .catch((err) => console.error("Failed to fetch indices", err));

    axios
      .get(`${API_BASE_URL}/api/screener`, { params: { sort_by: "market_cap", order: "desc", page: 1, page_size: 8 } })
      .then((res) => {
        setScreenerPreview(res.data.results);
        setTickerCount(res.data.total);
      })
      .catch((err) => console.error("Failed to fetch screener preview", err));
  }, []);

  const indexSectors = sectors.filter((s) => s.index_level != null);
  const realIndices = (indices ?? []).filter((i) => i.level != null);
  const cardsReady = indices !== null && sectors.length > 0;

  return (
    <div className="space-y-8">
      <Hero tickerCount={tickerCount} />

      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        {cardsReady ? (
          <>
            {realIndices.map((idx) => (
              <IndexCard key={idx.key} label={idx.name} value={idx.level} changePercent={idx.change_percent} />
            ))}
            {indexSectors.map((s) => (
              <IndexCard key={s.sector} label={s.sector} value={s.index_level} />
            ))}
          </>
        ) : (
          Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-24 col-span-full sm:col-span-1" />)
        )}
      </div>

      <MoversTable />

      <div>
        <h2 className="mb-3 text-lg font-bold">Curated Lists</h2>
        <CuratedLists />
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Largest Companies by Market Cap</CardTitle>
          <Button asChild variant="outline" size="sm">
            <Link to="/screener">View full screener</Link>
          </Button>
        </CardHeader>
        <CardContent>
          {screenerPreview ? <ScreenerTable rows={screenerPreview} /> : <Skeleton className="h-64 w-full" />}
        </CardContent>
      </Card>
    </div>
  );
}

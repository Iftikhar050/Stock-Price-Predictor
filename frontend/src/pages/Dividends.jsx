import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { ChevronUp, ChevronDown, ChevronsUpDown, Search } from "lucide-react";
import { API_BASE_URL } from "../config";
import { Card } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Skeleton } from "../components/ui/skeleton";
import { CompanyLogo } from "../components/CompanyLogo";
import { useSort } from "../lib/useSort";
import { cn } from "../lib/utils";

const FREQUENCIES = ["All", "Quarterly", "3x a Year", "Semi-Annual", "Annual", "Irregular"];

const COLUMNS = [
  { key: "symbol", label: "Symbol", align: "left" },
  { key: "name", label: "Company", align: "left", className: "hidden lg:table-cell" },
  { key: "dividend", label: "Dividend", align: "right", title: "Cash dividend per share over the last year of payouts, in Rs. (last 4 if quarterly, last 2 if semi-annual)" },
  { key: "dividend_yield", label: "Div. Yield", align: "right", title: "Annual dividend ÷ latest close" },
  { key: "dividend_growth", label: "Div. Growth", align: "right", title: "Annual dividend vs. the year of payouts before it" },
  { key: "ex_dividend_date", label: "Ex-Div Date", align: "right", title: "Most recent ex-dividend date" },
  { key: "payout_ratio", label: "Payout Ratio", align: "right", title: "Annual dividend ÷ sum of the last four reported quarterly EPS" },
  { key: "frequency", label: "Payout Freq.", align: "right", title: "Payouts per year over the last two years" },
];

const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(2)}%`);

const GROWTH_NOTES = {
  limited_history: { label: "Limited history", title: "Not enough earlier payouts on record to compare a full year" },
  resumed: { label: "Resumed", title: "Paid again after skipping the previous year" },
};

const PAYOUT_NOTES = {
  no_eps: { label: "No EPS data", title: "Recent quarterly EPS isn't available for this company" },
  loss: { label: "Loss-making", title: "Earnings over the last four quarters were zero or negative" },
  not_meaningful: { label: "n/m", title: "Over 300% of earnings - usually a one-off or special dividend" },
};

function Note({ note }) {
  return (
    <span className="text-xs text-muted-foreground" title={note.title}>
      {note.label}
    </span>
  );
}

function GrowthCell({ row }) {
  if (row.dividend_growth == null) return row.growth_note ? <Note note={GROWTH_NOTES[row.growth_note]} /> : "—";
  return pct(row.dividend_growth);
}

function PayoutCell({ row }) {
  if (row.payout_ratio == null) return row.payout_note ? <Note note={PAYOUT_NOTES[row.payout_note]} /> : "—";
  return pct(row.payout_ratio);
}

const formatDate = (iso) =>
  new Date(`${iso}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });

function HeaderCell({ col, sortKey, direction, onSort, className }) {
  const active = sortKey === col.key;
  const Icon = active ? (direction === "asc" ? ChevronUp : ChevronDown) : ChevronsUpDown;
  return (
    <th
      scope="col"
      title={col.title}
      className={cn("whitespace-nowrap px-3 py-3 font-semibold", col.className, className)}
      aria-sort={active ? (direction === "asc" ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        onClick={() => onSort(col.key)}
        className={cn(
          "inline-flex items-center gap-1 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-sm",
          col.align === "right" && "w-full justify-end"
        )}
      >
        {col.label}
        <Icon className={cn("h-3 w-3", !active && "text-muted-foreground/50")} />
      </button>
    </th>
  );
}

export function Dividends() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState("");
  const [frequency, setFrequency] = useState("All");

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/api/dividends`)
      .then((res) => setData(res.data))
      .catch((err) => {
        console.error("Failed to fetch dividends", err);
        setError("Couldn't load dividend data. Check that the API server is running, then refresh.");
      });
  }, []);

  const filtered = useMemo(() => {
    if (!data) return [];
    const q = query.trim().toLowerCase();
    return data.dividends.filter(
      (r) =>
        (frequency === "All" || r.frequency === frequency) &&
        (!q || r.symbol.toLowerCase().includes(q) || (r.name || "").toLowerCase().includes(q))
    );
  }, [data, query, frequency]);

  const { sortedRows, sortKey, direction, requestSort } = useSort(filtered, "dividend_yield", "desc");

  const nonPayers = useMemo(() => {
    if (!data?.non_payers || frequency !== "All") return [];
    const q = query.trim().toLowerCase();
    return data.non_payers.filter(
      (r) => !q || r.symbol.toLowerCase().includes(q) || (r.name || "").toLowerCase().includes(q)
    );
  }, [data, query, frequency]);

  if (error) {
    return <p className="rounded-lg border border-border bg-muted/40 p-4 text-sm text-muted-foreground">{error}</p>;
  }

  if (!data) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-12 w-64" />
        <Skeleton className="h-[480px] w-full" />
      </div>
    );
  }

  const stickyCell = "sticky left-0 z-10 bg-card";

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold">Dividends</h1>
        <p className="text-sm text-muted-foreground">
          {data.dividends.length} of {data.dividends.length + (data.non_payers?.length ?? 0)} companies paid a cash
          dividend in the last 15 months. Figures are per share, based on each company's last year of payouts, with
          prices as of {formatDate(data.as_of)}.
        </p>
      </div>

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap gap-2" role="group" aria-label="Filter by payout frequency">
          {FREQUENCIES.map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFrequency(f)}
              aria-pressed={frequency === f}
              className={cn(
                "rounded-full px-3.5 py-1.5 text-sm font-medium transition-colors",
                frequency === f ? "bg-foreground text-background" : "text-muted-foreground hover:bg-muted"
              )}
            >
              {f}
            </button>
          ))}
        </div>
        <div className="relative sm:w-64">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter by symbol or name"
            aria-label="Filter by symbol or company name"
            className="pl-8"
          />
        </div>
      </div>

      <Card className="overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-border text-left text-muted-foreground">
              <tr>
                <th scope="col" className={cn("w-12 whitespace-nowrap px-3 py-3 font-semibold", stickyCell)}>No.</th>
                {COLUMNS.map((col) => (
                  <HeaderCell
                    key={col.key}
                    col={col}
                    sortKey={sortKey}
                    direction={direction}
                    onSort={requestSort}
                    className={col.key === "symbol" ? cn(stickyCell, "left-12") : undefined}
                  />
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((r, i) => (
                <tr key={r.symbol} className="group border-b border-border last:border-0">
                  <td className={cn("px-3 py-2.5 tabular-nums text-muted-foreground group-hover:bg-muted", stickyCell)}>{i + 1}</td>
                  <td className={cn("px-3 py-2.5 group-hover:bg-muted", stickyCell, "left-12")}>
                    <Link to={`/company/${r.symbol}`} className="flex items-center gap-2 font-semibold text-primary hover:underline">
                      <CompanyLogo ticker={r.symbol} size="sm" />
                      {r.symbol}
                    </Link>
                  </td>
                  <td className="hidden max-w-[16rem] truncate px-3 py-2.5 text-muted-foreground group-hover:bg-muted/60 lg:table-cell" title={r.name}>
                    {r.name}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums group-hover:bg-muted/60">{r.dividend.toFixed(2)}</td>
                  <td className="px-3 py-2.5 text-right tabular-nums font-medium group-hover:bg-muted/60">{pct(r.dividend_yield)}</td>
                  <td
                    className={cn(
                      "px-3 py-2.5 text-right tabular-nums group-hover:bg-muted/60",
                      r.dividend_growth > 0 && "text-positive",
                      r.dividend_growth < 0 && "text-negative"
                    )}
                  >
                    <GrowthCell row={r} />
                  </td>
                  <td className="whitespace-nowrap px-3 py-2.5 text-right tabular-nums group-hover:bg-muted/60">{formatDate(r.ex_dividend_date)}</td>
                  <td className="px-3 py-2.5 text-right tabular-nums group-hover:bg-muted/60">
                    <PayoutCell row={r} />
                  </td>
                  <td className="whitespace-nowrap px-3 py-2.5 text-right group-hover:bg-muted/60">{r.frequency ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {sortedRows.length === 0 && (
            <p className="p-6 text-center text-sm text-muted-foreground">No companies match that filter.</p>
          )}
        </div>
      </Card>

      <p className="text-xs text-muted-foreground">
        Payout ratio uses the last four reported quarterly EPS figures. Special or one-off dividends count toward the
        year they were paid in, which can lift that year's yield and growth.
      </p>

      <NonPayers rows={nonPayers} />
    </div>
  );
}

function NonPayers({ rows }) {
  if (rows.length === 0) return null;
  return (
    <section className="space-y-3 pt-4" aria-labelledby="non-payers-title">
      <div>
        <h2 id="non-payers-title" className="text-lg font-semibold">
          No recent cash dividend <span className="text-muted-foreground">({rows.length})</span>
        </h2>
        <p className="text-sm text-muted-foreground">
          Companies with no cash dividend in the last 15 months, or none on record from our data source.
        </p>
      </div>
      <Card className="overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-border text-left text-muted-foreground">
              <tr>
                <th scope="col" className="px-3 py-3 font-semibold">Symbol</th>
                <th scope="col" className="hidden px-3 py-3 font-semibold sm:table-cell">Company</th>
                <th scope="col" className="px-3 py-3 text-right font-semibold">Last Dividend</th>
                <th scope="col" className="px-3 py-3 text-right font-semibold">Last Ex-Div Date</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.symbol} className="border-b border-border last:border-0 hover:bg-muted/60">
                  <td className="px-3 py-2.5">
                    <Link to={`/company/${r.symbol}`} className="flex items-center gap-2 font-semibold text-primary hover:underline">
                      <CompanyLogo ticker={r.symbol} size="sm" />
                      {r.symbol}
                    </Link>
                  </td>
                  <td className="hidden max-w-[18rem] truncate px-3 py-2.5 text-muted-foreground sm:table-cell" title={r.name}>
                    {r.name}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums">
                    {r.last_amount != null ? r.last_amount.toFixed(2) : <span className="text-xs text-muted-foreground">None on record</span>}
                  </td>
                  <td className="whitespace-nowrap px-3 py-2.5 text-right tabular-nums">
                    {r.last_ex_dividend_date ? formatDate(r.last_ex_dividend_date) : <span className="text-xs text-muted-foreground">—</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </section>
  );
}

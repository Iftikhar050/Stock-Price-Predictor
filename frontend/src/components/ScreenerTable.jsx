import React from "react";
import { Link } from "react-router-dom";
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell, SortableTableHead } from "./ui/table";
import { SignalBadge } from "./SignalBadge";
import { CompanyLogo } from "./CompanyLogo";
import { useSort } from "../lib/useSort";

function fmt(value, digits = 2) {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

function fmtCompact(value) {
  if (typeof value !== "number") return "—";
  if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  return value.toLocaleString();
}

const HEAD_CLASS = "py-2";
const CELL_CLASS = "py-1.5 tabular-nums";

// `predictions` is an optional { [ticker]: predictionResponse } map. Omitted by
// default (Home/Sectors/etc. never fetch it) - only the Screener page's opt-in
// "Run ML Predictions" action populates it, scoped to the rows on screen.
export function ScreenerTable({ rows, predictions = null, predictionsLoading = false }) {
  const { sortedRows, sortKey, direction, requestSort } = useSort(rows, null, "desc");

  if (!rows || rows.length === 0) {
    return <p className="py-8 text-center text-sm text-muted-foreground">No companies match these filters.</p>;
  }

  const showSignalColumn = predictions != null || predictionsLoading;
  const sortHeadProps = { activeKey: sortKey, direction, onSort: requestSort, className: HEAD_CLASS };

  return (
    <Table>
      <TableHeader>
        <TableRow className="hover:bg-transparent">
          <SortableTableHead label="Company" sortKey="ticker" {...sortHeadProps} />
          <SortableTableHead label="Price" sortKey="price" align="right" {...sortHeadProps} />
          <SortableTableHead label="Change" sortKey="change_percent" align="right" {...sortHeadProps} />
          <SortableTableHead label="Volume" sortKey="volume" align="right" {...sortHeadProps} />
          <SortableTableHead label="Mkt Cap" sortKey="market_cap" align="right" {...sortHeadProps} />
          <SortableTableHead label="P/E" sortKey="pe_ratio" align="right" {...sortHeadProps} />
          <SortableTableHead label="P/B" sortKey="pb_ratio" align="right" {...sortHeadProps} />
          <SortableTableHead label="Div Yld" sortKey="dividend_yield" align="right" {...sortHeadProps} />
          <SortableTableHead label="ROE" sortKey="roe" align="right" {...sortHeadProps} />
          {showSignalColumn && (
            <TableHead
              className={`${HEAD_CLASS} text-right`}
              title="Experimental: the underlying model's directional accuracy has not been validated above chance. Not investment advice."
            >
              Signal
            </TableHead>
          )}
        </TableRow>
      </TableHeader>
      <TableBody>
        {sortedRows.map((row) => (
          <TableRow key={row.ticker}>
            <TableCell className={CELL_CLASS}>
              <Link to={`/company/${row.ticker}`} className="flex items-center gap-3 hover:underline">
                <CompanyLogo ticker={row.ticker} size="md" />
                <div className="min-w-0">
                  <div className="font-semibold text-primary">{row.ticker}</div>
                  <div className="truncate text-xs text-muted-foreground">
                    {row.name}{row.sector ? ` · ${row.sector}` : ""}
                  </div>
                </div>
              </Link>
            </TableCell>
            <TableCell className={`${CELL_CLASS} text-right font-medium`}>{fmt(row.price)}</TableCell>
            <TableCell className={`${CELL_CLASS} text-right ${row.change_percent >= 0 ? "text-positive" : "text-negative"}`}>
              {row.change_percent >= 0 ? "▲" : "▼"} {fmt(Math.abs(row.change_percent))}%
            </TableCell>
            <TableCell className={`${CELL_CLASS} text-right text-muted-foreground`}>{fmtCompact(row.volume)}</TableCell>
            <TableCell className={`${CELL_CLASS} text-right text-muted-foreground`}>{fmtCompact(row.market_cap)}</TableCell>
            <TableCell className={`${CELL_CLASS} text-right`}>{fmt(row.pe_ratio)}</TableCell>
            <TableCell className={`${CELL_CLASS} text-right`}>{fmt(row.pb_ratio)}</TableCell>
            <TableCell className={`${CELL_CLASS} text-right`}>{row.dividend_yield != null ? `${(row.dividend_yield * 100).toFixed(2)}%` : "—"}</TableCell>
            <TableCell className={`${CELL_CLASS} text-right`}>{row.roe != null ? `${(row.roe * 100).toFixed(2)}%` : "—"}</TableCell>
            {showSignalColumn && (
              <TableCell className={`${CELL_CLASS} text-right`}>
                {predictions?.[row.ticker] ? (
                  <SignalBadge prediction={predictions[row.ticker]} />
                ) : (
                  <span className="text-xs text-muted-foreground">{predictionsLoading ? "…" : "—"}</span>
                )}
              </TableCell>
            )}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

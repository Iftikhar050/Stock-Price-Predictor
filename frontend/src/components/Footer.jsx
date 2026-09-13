import React from "react";
import { Link } from "react-router-dom";
import { LineChart, TrendingUp, LayoutGrid, GitCompare, LayoutList, ShieldAlert } from "lucide-react";

const PLATFORM_LINKS = [
  { to: "/", label: "Home" },
  { to: "/screener", label: "Screener" },
  { to: "/compare", label: "Compare" },
  { to: "/sectors", label: "Sectors" },
];

const FEATURES = [
  "4-Model AI Price Predictions",
  "Real-Time Market Data",
  "Fundamental Screener",
  "Sector Benchmarking",
  "Dividend & Events Tracking",
];

const COVERAGE = [
  "KSE-100 Index companies",
  "Daily price & volume history",
  "Quarterly fundamentals",
  "Corporate announcements",
];

const COLUMN_TITLE = "flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground";
const LINK_CLASS =
  "group inline-flex items-center gap-1 text-muted-foreground transition-colors hover:text-foreground";
const DOT = <span className="h-1 w-1 shrink-0 rounded-full bg-primary/50" aria-hidden="true" />;

export function Footer() {
  return (
    <footer className="relative mt-10 pb-20 md:pb-0">
      <div className="h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />
      <div className="border-t border-border">
        <div className="mx-auto grid max-w-7xl grid-cols-1 gap-10 px-4 py-12 sm:grid-cols-2 lg:grid-cols-4">
          <div className="sm:col-span-2 lg:col-span-1">
            <Link to="/" className="flex items-center gap-2 font-semibold text-foreground">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
                <LineChart className="h-4.5 w-4.5" />
              </span>
              Stock Analyst
            </Link>
            <p className="mt-3 max-w-xs text-sm leading-relaxed text-muted-foreground">
              Real-time prices, fundamentals, and a 4-model machine learning price signal for every KSE-100 company on
              the Pakistan Stock Exchange.
            </p>
          </div>

          <div>
            <h3 className={COLUMN_TITLE}>
              <LayoutGrid className="h-3.5 w-3.5" />
              Platform
            </h3>
            <ul className="mt-4 space-y-2.5 text-sm">
              {PLATFORM_LINKS.map((l) => (
                <li key={l.to}>
                  <Link to={l.to} className={LINK_CLASS}>
                    {DOT}
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className={COLUMN_TITLE}>
              <TrendingUp className="h-3.5 w-3.5" />
              Features
            </h3>
            <ul className="mt-4 space-y-2.5 text-sm text-muted-foreground">
              {FEATURES.map((f) => (
                <li key={f} className="flex items-center gap-1.5">
                  {DOT}
                  {f}
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className={COLUMN_TITLE}>
              <LayoutList className="h-3.5 w-3.5" />
              Coverage
            </h3>
            <ul className="mt-4 space-y-2.5 text-sm text-muted-foreground">
              {COVERAGE.map((c) => (
                <li key={c} className="flex items-center gap-1.5">
                  {DOT}
                  {c}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div className="border-t border-border">
        <div className="mx-auto max-w-7xl px-4 py-6">
          <div className="flex items-start gap-3 rounded-lg border border-border bg-muted/40 p-4">
            <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <p className="text-xs leading-relaxed text-muted-foreground">
              <span className="font-semibold text-foreground">Disclaimer: </span>
              All data, analytics, and model predictions on this platform are provided for informational and research
              purposes only and do not constitute investment, financial, or trading advice. Predictions are
              statistical estimates and are not guarantees of future performance. Verify figures against official PSX
              sources before making any investment decision.
            </p>
          </div>
          <div className="mt-5 flex flex-col gap-2 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
            <span>© {new Date().getFullYear()} Stock Analyst. All rights reserved.</span>
            <span className="flex items-center gap-1.5">
              <span className="h-1 w-1 rounded-full bg-positive" aria-hidden="true" />
              Data sourced from the Pakistan Stock Exchange and the State Bank of Pakistan.
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}

import React, { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { useNavigate } from "react-router-dom";
import { Search } from "lucide-react";
import { CompanyLogo } from "./CompanyLogo";

const MAX_RESULTS = 8;

export function CommandPalette({ open, onClose, tickers }) {
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIndex(0);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return tickers.slice(0, MAX_RESULTS);
    return tickers
      .filter((t) => t.ticker.toLowerCase().includes(q) || t.name.toLowerCase().includes(q))
      .slice(0, MAX_RESULTS);
  }, [query, tickers]);

  const select = (t) => {
    if (!t) return;
    navigate(`/company/${t.ticker}`);
    onClose();
  };

  const handleKeyDown = (e) => {
    if (e.key === "Escape") {
      onClose();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      select(results[activeIndex]);
    }
  };

  if (!open) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex justify-center bg-black/50 px-4 pt-[12vh]"
      onClick={onClose}
    >
      <div
        className="h-fit w-full max-w-md overflow-hidden rounded-xl border border-border bg-card shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2.5 border-b border-border px-3 py-2.5">
          <Search className="h-4 w-4 shrink-0 text-muted-foreground" />
          <input
            autoFocus
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActiveIndex(0);
            }}
            onKeyDown={handleKeyDown}
            placeholder="Search companies..."
            className="w-full min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
          <kbd className="shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
            ESC
          </kbd>
        </div>
        <div className="max-h-80 overflow-y-auto py-1">
          {results.length === 0 ? (
            <p className="px-4 py-5 text-center text-sm text-muted-foreground">No companies found.</p>
          ) : (
            results.map((t, idx) => (
              <button
                key={t.ticker}
                type="button"
                onClick={() => select(t)}
                onMouseEnter={() => setActiveIndex(idx)}
                className={`flex w-full items-center gap-2.5 px-3 py-2 text-left transition-colors ${
                  idx === activeIndex ? "bg-muted" : ""
                }`}
              >
                <CompanyLogo ticker={t.ticker} size="sm" />
                <span className="text-sm font-semibold">{t.ticker}</span>
                <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground">{t.name}</span>
                {t.sector && (
                  <span className="hidden shrink-0 text-[11px] text-muted-foreground sm:block">{t.sector}</span>
                )}
              </button>
            ))
          )}
        </div>
      </div>
    </div>,
    document.body
  );
}

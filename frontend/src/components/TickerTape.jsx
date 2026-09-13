import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { API_BASE_URL } from "../config";
import { CompanyLogo } from "./CompanyLogo";

function TapeItem({ item }) {
  const up = item.change_percent >= 0;
  return (
    <Link
      to={`/company/${item.symbol}`}
      className="flex shrink-0 items-center gap-2 border-r border-border px-4 py-2 text-sm hover:bg-muted/50"
    >
      <CompanyLogo ticker={item.symbol} size="sm" />
      <span className="font-semibold">{item.symbol}</span>
      <span className="tabular-nums text-muted-foreground">{item.price.toFixed(2)}</span>
      <span className={`flex items-center gap-0.5 tabular-nums font-medium ${up ? "text-positive" : "text-negative"}`}>
        {up ? "▲" : "▼"} {Math.abs(item.change_percent).toFixed(2)}%
      </span>
    </Link>
  );
}

export function TickerTape() {
  const [items, setItems] = useState([]);

  useEffect(() => {
    const fetchTape = () => {
      axios
        .get(`${API_BASE_URL}/api/market_performers`)
        .then((res) => setItems(res.data.top_active?.slice(0, 15) ?? []))
        .catch((err) => console.error("Failed to fetch ticker tape", err));
    };
    fetchTape();
    const interval = setInterval(fetchTape, 30000);
    return () => clearInterval(interval);
  }, []);

  if (items.length === 0) return null;

  return (
    <div className="border-b border-border bg-card/60 overflow-hidden">
      <div className="ticker-tape-track flex w-max">
        {/* duplicated once for a seamless loop - animation scrolls exactly -50% */}
        {[...items, ...items].map((item, idx) => (
          <TapeItem key={`${item.symbol}-${idx}`} item={item} />
        ))}
      </div>
    </div>
  );
}

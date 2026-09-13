import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { ChevronUp, ChevronDown, ChevronsUpDown } from "lucide-react";
import { API_BASE_URL } from "../config";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Skeleton } from "./ui/skeleton";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./ui/tabs";
import { CompanyLogo } from "./CompanyLogo";
import { useSort } from "../lib/useSort";

const TABS = [
  { key: "top_active", label: "Most Active" },
  { key: "top_advancers", label: "Top Advancers" },
  { key: "top_decliners", label: "Top Decliners" },
];

const ROW_GRID = "grid grid-cols-[1.5fr_1fr_1.2fr_1.5fr] items-center gap-2";

const MOVERS_COLUMNS = [
  { key: "symbol", label: "Symbol", align: "left" },
  { key: "price", label: "Price", align: "right" },
  { key: "change_percent", label: "Change", align: "right" },
  { key: "volume", label: "Volume", align: "right" },
];

function MoversHeader({ activeKey, direction, onSort }) {
  return (
    <div className={`${ROW_GRID} border-b border-border pb-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground`}>
      {MOVERS_COLUMNS.map((c) => {
        const isActive = activeKey === c.key;
        const Icon = isActive ? (direction === "asc" ? ChevronUp : ChevronDown) : ChevronsUpDown;
        return (
          <button
            key={c.key}
            onClick={() => onSort(c.key)}
            className={`flex cursor-pointer select-none items-center gap-1 hover:text-foreground ${c.align === "right" ? "justify-end" : ""}`}
          >
            {c.label}
            <Icon className={`h-3 w-3 ${!isActive ? "text-muted-foreground/50" : ""}`} />
          </button>
        );
      })}
    </div>
  );
}

function Row({ item }) {
  return (
    <Link
      to={`/company/${item.symbol}`}
      className={`${ROW_GRID} border-b border-border py-2 text-sm last:border-0 hover:bg-muted/50 transition-colors`}
    >
      <span className="flex items-center gap-2 font-semibold text-primary">
        <CompanyLogo ticker={item.symbol} size="sm" />
        {item.symbol}
      </span>
      <span className="tabular-nums text-right">{item.price.toFixed(2)}</span>
      <span className={`tabular-nums text-right ${item.change >= 0 ? "text-positive" : "text-negative"}`}>
        {item.change >= 0 ? "▲" : "▼"} {Math.abs(item.change_percent).toFixed(2)}%
      </span>
      <span className="tabular-nums text-right text-muted-foreground">{item.volume.toLocaleString()}</span>
    </Link>
  );
}

function MoversList({ items, limit }) {
  const { sortedRows, sortKey, direction, requestSort } = useSort(items.slice(0, limit), null, "desc");
  return (
    <>
      <MoversHeader activeKey={sortKey} direction={direction} onSort={requestSort} />
      {sortedRows.map((item) => (
        <Row key={item.symbol} item={item} />
      ))}
    </>
  );
}

export function MoversTable({ limit = 8 }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    const fetchPerformers = () => {
      axios
        .get(`${API_BASE_URL}/api/market_performers`)
        .then((res) => setData(res.data))
        .catch((err) => console.error("Failed to fetch market performers", err));
    };
    fetchPerformers();
    const interval = setInterval(fetchPerformers, 30000);
    return () => clearInterval(interval);
  }, []);

  if (!data) {
    return <Skeleton className="h-72 w-full" />;
  }

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Market Movers</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="top_active">
          <TabsList>
            {TABS.map((t) => (
              <TabsTrigger key={t.key} value={t.key}>{t.label}</TabsTrigger>
            ))}
          </TabsList>
          {TABS.map((t) => (
            <TabsContent key={t.key} value={t.key}>
              <MoversList items={data[t.key]} limit={limit} />
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  );
}

import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { API_BASE_URL } from "../config";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Skeleton } from "./ui/skeleton";
import { CompanyLogo } from "./CompanyLogo";

const LISTS = [
  { key: "dividend_yield", title: "Top Dividend Yield", sort_by: "dividend_yield", fmt: (v) => `${(v * 100).toFixed(2)}%` },
  { key: "roe", title: "Highest ROE", sort_by: "roe", fmt: (v) => `${(v * 100).toFixed(2)}%` },
  { key: "change_percent", title: "Top Gainers Today", sort_by: "change_percent", fmt: (v) => `+${v.toFixed(2)}%` },
];

export function CuratedLists() {
  const [data, setData] = useState(null);

  useEffect(() => {
    // These 3 lists used to be 3 separate /api/screener requests, each
    // sorting the exact same underlying row set by a different field. One
    // fetch of every active ticker + a client-side sort per list gets the
    // same result with a third of the round trips.
    axios
      .get(`${API_BASE_URL}/api/screener`, { params: { sort_by: "market_cap", order: "desc", page: 1, page_size: 200 } })
      .then((res) => {
        const rows = res.data.results;
        setData(Object.fromEntries(
          LISTS.map((l) => [
            l.key,
            [...rows]
              .filter((r) => r[l.key] != null)
              .sort((a, b) => b[l.key] - a[l.key])
              .slice(0, 5),
          ])
        ));
      })
      .catch(() => setData(Object.fromEntries(LISTS.map((l) => [l.key, []]))));
  }, []);

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
      {LISTS.map((list) => (
        <Card key={list.key} className="transition-shadow hover:shadow-md">
          <CardHeader>
            <CardTitle className="text-base">{list.title}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            {!data ? (
              Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-7 w-full" />)
            ) : data[list.key].length === 0 ? (
              <p className="py-2 text-sm text-muted-foreground">No data available.</p>
            ) : (
              data[list.key].map((row) => {
                const value = row[list.key];
                return (
                  <Link
                    key={row.ticker}
                    to={`/company/${row.ticker}`}
                    className="flex items-center justify-between rounded-md px-1.5 py-1.5 text-sm transition-colors hover:bg-muted"
                  >
                    <span className="flex items-center gap-2 font-semibold text-primary">
                      <CompanyLogo ticker={row.ticker} size="sm" />
                      {row.ticker}
                    </span>
                    <span className="tabular-nums text-positive">{typeof value === "number" ? list.fmt(value) : "—"}</span>
                  </Link>
                );
              })
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

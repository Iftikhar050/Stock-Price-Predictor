import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { API_BASE_URL } from "../config";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Skeleton } from "../components/ui/skeleton";

export function Sectors() {
  const [sectors, setSectors] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/api/sectors`)
      .then((res) => setSectors(res.data.sectors))
      .catch((err) => console.error("Failed to fetch sectors", err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-36 w-full" />)}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold">Sectors</h1>
        <p className="text-sm text-muted-foreground">PSX-published sector indices where available, peer-average valuations everywhere else.</p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {sectors.map((s) => (
          <Link key={s.sector} to={`/sectors/${encodeURIComponent(s.sector)}`}>
            <Card className="h-full shadow-sm transition-all hover:border-primary/30 hover:shadow-md">
              <CardHeader>
                <CardTitle>{s.sector}</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <div className="text-xs text-muted-foreground">Companies</div>
                  <div className="font-semibold tabular-nums">{s.ticker_count}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Index Level</div>
                  <div className="font-semibold tabular-nums">{s.index_level != null ? s.index_level.toLocaleString() : "—"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Avg P/E</div>
                  <div className="font-semibold tabular-nums">{s.avg_pe != null ? s.avg_pe.toFixed(2) : "—"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Avg P/B</div>
                  <div className="font-semibold tabular-nums">{s.avg_pb != null ? s.avg_pb.toFixed(2) : "—"}</div>
                </div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}

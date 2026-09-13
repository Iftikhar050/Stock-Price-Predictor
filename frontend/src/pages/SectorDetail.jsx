import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import axios from "axios";
import { ArrowLeft } from "lucide-react";
import { API_BASE_URL } from "../config";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Skeleton } from "../components/ui/skeleton";
import { ScreenerTable } from "../components/ScreenerTable";

export function SectorDetail() {
  const { sector } = useParams();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    axios
      .get(`${API_BASE_URL}/api/sectors/${encodeURIComponent(sector)}`)
      .then((res) => setData(res.data))
      .catch((err) => setError(err.response?.status === 404 ? "Sector not found." : "Failed to load sector."))
      .finally(() => setLoading(false));
  }, [sector]);

  return (
    <div className="space-y-4">
      <Link to="/sectors" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="h-4 w-4" /> All sectors
      </Link>
      <h1 className="text-2xl font-bold">{sector}</h1>

      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle>{data ? `${data.tickers.length} companies` : "Loading..."}</CardTitle>
        </CardHeader>
        <CardContent>
          {loading && <Skeleton className="h-64 w-full" />}
          {error && <p className="py-8 text-center text-sm text-negative">{error}</p>}
          {data && <ScreenerTable rows={data.tickers} />}
        </CardContent>
      </Card>
    </div>
  );
}

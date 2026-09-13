import React, { useState } from "react";
import { TickerAvatar } from "./TickerAvatar";
import logoManifest from "../data/logoManifest.json";

const SIZE_PX = { sm: 24, md: 36, lg: 56 };

// Bulk-downloading logos from aggregator sites (brandlogos.net, seeklogo.com)
// risks wrong matches across ~100 tickers and murky redistribution rights.
// Instead, scratch/fetch_logos.py pulled each ticker's real favicon/logo
// from its own verified website domain (already scraped into our DB via
// /api/company/{ticker}) and saved it locally under public/logos/, indexed
// by logoManifest.json - so lookups here are local and don't depend on a
// live third-party service. Falls back to a live per-domain fetch (for any
// ticker fetched after the manifest was built), then to the colored
// initials avatar if nothing is available.
export function CompanyLogo({ ticker, domain, size = "md" }) {
  const [failed, setFailed] = useState(false);
  const px = SIZE_PX[size];
  const localSrc = ticker ? logoManifest[ticker.toUpperCase()] : null;
  const src = localSrc || (domain ? `https://icons.duckduckgo.com/ip3/${domain}.ico` : null);

  if (!src || failed) return <TickerAvatar ticker={ticker} size={size} />;

  return (
    <img
      src={src}
      alt={`${ticker} logo`}
      className="shrink-0 rounded-full border border-border bg-white object-contain p-1"
      style={{ width: px, height: px }}
      onError={() => setFailed(true)}
    />
  );
}

export function cleanDomain(url) {
  if (!url) return null;
  return url.replace(/^https?:\/\//, "").replace(/^www\./, "").split("/")[0].trim() || null;
}

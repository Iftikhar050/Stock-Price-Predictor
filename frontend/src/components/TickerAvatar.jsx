import React from "react";
import { cn } from "../lib/utils";

// No real logo assets exist for these companies, so each ticker gets a
// deterministic colored initial-avatar (same idea as Slack/GitHub avatars)
// instead of a fabricated brand mark - same ticker always gets the same color.
function hueForTicker(ticker) {
  let hash = 0;
  for (let i = 0; i < ticker.length; i++) {
    hash = (hash * 31 + ticker.charCodeAt(i)) % 360;
  }
  return hash < 0 ? hash + 360 : hash;
}

const SIZE_CLASSES = {
  sm: "h-6 w-6 text-[10px]",
  md: "h-9 w-9 text-xs",
  lg: "h-14 w-14 text-lg",
};

export function TickerAvatar({ ticker, size = "md", className }) {
  const hue = hueForTicker(ticker);
  const initials = ticker.slice(0, 2).toUpperCase();
  return (
    <div
      className={cn(
        "flex shrink-0 items-center justify-center rounded-full font-bold text-white",
        SIZE_CLASSES[size],
        className
      )}
      style={{ backgroundColor: `hsl(${hue}, 65%, 42%)` }}
    >
      {initials}
    </div>
  );
}

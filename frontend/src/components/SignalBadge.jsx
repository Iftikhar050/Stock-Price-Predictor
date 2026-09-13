import React from "react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { Badge } from "./ui/badge";
import { computeSignal } from "../lib/signal";

const CONFIG = {
  bullish: { variant: "positive", label: "Bullish", Icon: TrendingUp },
  bearish: { variant: "negative", label: "Bearish", Icon: TrendingDown },
  neutral: { variant: "muted", label: "Neutral", Icon: Minus },
};

export function SignalBadge({ prediction, className }) {
  const signal = computeSignal(prediction);
  if (!signal) return null;

  const { variant, label, Icon } = CONFIG[signal.direction];
  const confidenceText = signal.confidence != null ? ` · ${signal.confidence.toFixed(0)}%` : "";

  return (
    <Badge variant={variant} className={className}>
      <Icon className="mr-1 h-3 w-3" />
      {label}
      {confidenceText}
    </Badge>
  );
}

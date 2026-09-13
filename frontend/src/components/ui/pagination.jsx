import React from "react";
import { cn } from "../../lib/utils";

export function PaginationControls({ page, totalPages, hasPrev, hasNext, onPrev, onNext, className }) {
  if (totalPages <= 1) return null;
  return (
    <div className={cn("flex items-center justify-between border-t border-border pt-3", className)}>
      <span className="text-xs text-muted-foreground">Page {page} of {totalPages}</span>
      <div className="flex items-center gap-2">
        <button
          onClick={onPrev}
          disabled={!hasPrev}
          className="rounded-md border border-border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-40"
        >
          Previous
        </button>
        <button
          onClick={onNext}
          disabled={!hasNext}
          className="rounded-md border border-border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </div>
  );
}

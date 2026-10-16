import { useEffect, useMemo, useState } from "react";

// Simple client-side pagination for an already-loaded array. Resets to page 1
// whenever the underlying array reference changes (a new fetch, a re-sort,
// or switching to a different ticker's data).
export function usePagination(items, pageSize = 5) {
  const [page, setPage] = useState(1);

  useEffect(() => setPage(1), [items]);

  const total = items?.length ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const safePage = Math.min(page, totalPages);

  const pageItems = useMemo(() => {
    const start = (safePage - 1) * pageSize;
    return (items ?? []).slice(start, start + pageSize);
  }, [items, safePage, pageSize]);

  return {
    pageItems,
    page: safePage,
    totalPages,
    hasPrev: safePage > 1,
    hasNext: safePage < totalPages,
    goPrev: () => setPage((p) => Math.max(1, p - 1)),
    goNext: () => setPage((p) => Math.min(totalPages, p + 1)),
  };
}

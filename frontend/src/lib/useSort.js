import { useMemo, useState } from "react";

// Generic client-side sort for an array of row objects by a given key.
// Strings sort locale-aware; numbers/booleans/dates (already comparable
// primitives) sort by value; nullish values always sort to the bottom
// regardless of direction, since "unknown" isn't meaningfully high or low.
export function useSort(rows, initialKey = null, initialDirection = "asc") {
  const [sortKey, setSortKey] = useState(initialKey);
  const [direction, setDirection] = useState(initialDirection);

  const requestSort = (key) => {
    if (sortKey === key) {
      setDirection((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setDirection("asc");
    }
  };

  const sortedRows = useMemo(() => {
    if (!sortKey || !rows) return rows;
    const copy = [...rows];
    copy.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "string" && typeof bv === "string") {
        return direction === "asc" ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      if (av < bv) return direction === "asc" ? -1 : 1;
      if (av > bv) return direction === "asc" ? 1 : -1;
      return 0;
    });
    return copy;
  }, [rows, sortKey, direction]);

  return { sortedRows, sortKey, direction, requestSort };
}

import React from "react";
import { ChevronUp, ChevronDown, ChevronsUpDown } from "lucide-react";
import { cn } from "../../lib/utils";

export const Table = React.forwardRef(({ className, ...props }, ref) => (
  <div className="w-full overflow-x-auto rounded-lg border border-border">
    <table ref={ref} className={cn("w-full caption-bottom text-sm", className)} {...props} />
  </div>
));
Table.displayName = "Table";

export const TableHeader = React.forwardRef(({ className, ...props }, ref) => (
  <thead ref={ref} className={cn("bg-muted/60 [&_tr]:border-b border-border", className)} {...props} />
));
TableHeader.displayName = "TableHeader";

export const TableBody = React.forwardRef(({ className, ...props }, ref) => (
  <tbody ref={ref} className={cn("[&_tr:last-child]:border-0", className)} {...props} />
));
TableBody.displayName = "TableBody";

export const TableRow = React.forwardRef(({ className, ...props }, ref) => (
  <tr
    ref={ref}
    className={cn("border-b border-border transition-colors hover:bg-muted/40", className)}
    {...props}
  />
));
TableRow.displayName = "TableRow";

export const TableHead = React.forwardRef(({ className, ...props }, ref) => (
  <th
    ref={ref}
    className={cn(
      "h-9 px-3 text-left align-middle text-xs font-semibold uppercase tracking-wide text-muted-foreground",
      className
    )}
    {...props}
  />
));
TableHead.displayName = "TableHead";

export const TableCell = React.forwardRef(({ className, ...props }, ref) => (
  <td ref={ref} className={cn("px-3 py-2.5 align-middle", className)} {...props} />
));
TableCell.displayName = "TableCell";

// Clickable column header for client-side sortable tables. `sortKey` is the
// row-object key this column sorts by; `activeKey`/`direction` come from the
// useSort() hook and decide which chevron state to render.
export const SortableTableHead = React.forwardRef(
  ({ label, sortKey, activeKey, direction, onSort, align = "left", className, ...props }, ref) => {
    const isActive = activeKey === sortKey;
    const Icon = isActive ? (direction === "asc" ? ChevronUp : ChevronDown) : ChevronsUpDown;
    return (
      <TableHead
        ref={ref}
        onClick={() => onSort(sortKey)}
        className={cn("cursor-pointer select-none hover:text-foreground", align === "right" && "text-right", className)}
        {...props}
      >
        <span className={cn("inline-flex items-center gap-1", align === "right" && "flex-row-reverse")}>
          {label}
          <Icon className={cn("h-3.5 w-3.5", !isActive && "text-muted-foreground/50")} />
        </span>
      </TableHead>
    );
  }
);
SortableTableHead.displayName = "SortableTableHead";

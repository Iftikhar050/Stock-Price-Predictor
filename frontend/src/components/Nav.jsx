import React, { useEffect, useState } from "react";
import { NavLink, Link, useLocation } from "react-router-dom";
import axios from "axios";
import { LineChart, Search, ChevronDown } from "lucide-react";
import { API_BASE_URL } from "../config";
import { ThemeToggle } from "./ThemeToggle";
import { CommandPalette } from "./CommandPalette";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem } from "./ui/dropdown-menu";
import { MARKET_TABS } from "../pages/Markets";
import { cn } from "../lib/utils";

const NAV_LINKS = [
  { to: "/", label: "Home", end: true },
  { to: "/screener", label: "Screener" },
  { to: "/compare", label: "Compare" },
  { to: "/sectors", label: "Sectors" },
];

function NavItem({ to, label, end }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        cn(
          "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
          isActive ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground"
        )
      }
    >
      {label}
    </NavLink>
  );
}

function MarketsDropdown() {
  const location = useLocation();
  const active = location.pathname.startsWith("/markets");

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        className={cn(
          "inline-flex items-center gap-1 rounded-md px-3 py-1.5 text-sm font-medium outline-none transition-colors",
          active ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground"
        )}
      >
        Markets <ChevronDown className="h-3.5 w-3.5" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {MARKET_TABS.map((t) => (
          <DropdownMenuItem key={t.key} asChild>
            <Link to={`/markets/${t.key}`}>{t.label}</Link>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function Nav() {
  const [tickers, setTickers] = useState([]);
  const [searchOpen, setSearchOpen] = useState(false);

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/api/tickers`)
      .then((res) => setTickers(res.data))
      .catch((err) => console.error("Failed to fetch tickers", err));
  }, []);

  useEffect(() => {
    const handler = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setSearchOpen(true);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <header className="sticky top-0 z-40 border-b border-border bg-card/80 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3">
        <NavLink to="/" className="flex shrink-0 items-center gap-2 font-semibold text-foreground">
          <LineChart className="h-5 w-5 text-primary" />
          <span className="hidden sm:inline">Stock Analyst</span>
        </NavLink>

        <button
          type="button"
          onClick={() => setSearchOpen(true)}
          className="flex h-9 flex-1 items-center gap-2 rounded-lg border border-input bg-muted/50 px-3 text-sm text-muted-foreground transition-colors hover:bg-muted sm:max-w-sm"
        >
          <Search className="h-4 w-4 shrink-0" />
          <span className="flex-1 truncate text-left">Search companies...</span>
          <kbd className="hidden shrink-0 rounded border border-border bg-card px-1.5 py-0.5 text-[10px] font-medium sm:block">
            ⌘K
          </kbd>
        </button>

        <nav className="ml-auto hidden items-center gap-1 md:flex">
          {NAV_LINKS.slice(0, 2).map((link) => <NavItem key={link.to} {...link} />)}
          <MarketsDropdown />
          {NAV_LINKS.slice(2).map((link) => <NavItem key={link.to} {...link} />)}
        </nav>

        <div className="ml-auto md:ml-0">
          <ThemeToggle />
        </div>
      </div>

      <CommandPalette open={searchOpen} onClose={() => setSearchOpen(false)} tickers={tickers} />
    </header>
  );
}

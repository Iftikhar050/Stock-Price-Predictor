import React from "react";
import { NavLink } from "react-router-dom";
import { Home, SlidersHorizontal, Landmark, GitCompare, Layers } from "lucide-react";
import { cn } from "../lib/utils";

const TABS = [
  { to: "/", label: "Home", end: true, icon: Home },
  { to: "/screener", label: "Screener", icon: SlidersHorizontal },
  { to: "/markets", label: "Markets", icon: Landmark },
  { to: "/compare", label: "Compare", icon: GitCompare },
  { to: "/sectors", label: "Sectors", icon: Layers },
];

export function MobileTabBar() {
  return (
    <nav className="fixed inset-x-0 bottom-0 z-40 flex border-t border-border bg-card/95 backdrop-blur md:hidden">
      {TABS.map(({ to, label, end, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          className={({ isActive }) =>
            cn(
              "flex flex-1 flex-col items-center gap-0.5 py-2 text-xs font-medium transition-colors",
              isActive ? "text-primary" : "text-muted-foreground"
            )
          }
        >
          <Icon className="h-5 w-5" />
          {label}
        </NavLink>
      ))}
    </nav>
  );
}

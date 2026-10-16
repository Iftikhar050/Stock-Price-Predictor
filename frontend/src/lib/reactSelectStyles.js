// react-select ships its own inline styles by default; these overrides key off
// our CSS custom properties (index.css) so the combobox follows the light/dark
// toggle automatically without needing to read theme state in JS.
export const selectStyles = {
  control: (base, state) => ({
    ...base,
    backgroundColor: "var(--card)",
    borderColor: state.isFocused ? "var(--primary)" : "var(--border)",
    boxShadow: state.isFocused ? "0 0 0 2px var(--ring)" : "none",
    minHeight: "2.25rem",
    "&:hover": { borderColor: "var(--primary)" },
  }),
  input: (base) => ({ ...base, color: "var(--foreground)" }),
  singleValue: (base) => ({ ...base, color: "var(--foreground)" }),
  placeholder: (base) => ({ ...base, color: "var(--muted-foreground)" }),
  menu: (base) => ({ ...base, backgroundColor: "var(--card)", border: "1px solid var(--border)", zIndex: 50 }),
  option: (base, state) => ({
    ...base,
    backgroundColor: state.isSelected ? "var(--primary)" : state.isFocused ? "var(--muted)" : "transparent",
    color: state.isSelected ? "var(--primary-foreground)" : "var(--foreground)",
    cursor: "pointer",
  }),
  indicatorSeparator: (base) => ({ ...base, backgroundColor: "var(--border)" }),
  dropdownIndicator: (base) => ({ ...base, color: "var(--muted-foreground)" }),
  clearIndicator: (base) => ({ ...base, color: "var(--muted-foreground)" }),
};

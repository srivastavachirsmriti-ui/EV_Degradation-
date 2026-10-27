// Small, labelled navigation symbols support recognition without relying on color.
const paths = {
  battery: "M3 7h16v10H3z M21 10v4 M7 10v4 M11 10v4 M15 10v4",
  overview: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
  classify: "M21 11a8 8 0 0 1-8 8H6l-4 3V11a9 9 0 0 1 18-4 M7 9h9 M7 13h6",
  estimate: "M4 19a9 9 0 1 1 16 0 M12 15l5-6 M6 19h12",
  results: "M12 3l8 3v6c0 5-8 9-8 9s-8-4-8-9V6z M8 12l3 3 5-6",
  records: "M5 3h14v18H5z M9 8h6 M9 12h6 M9 16h4",
  arrow: "M5 12h14 M14 7l5 5-5 5",
  info: "M12 8h.01 M12 11v5 M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0",
};
export default function Icon({ name, size = 20 }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name] || paths.info} /></svg>;
}

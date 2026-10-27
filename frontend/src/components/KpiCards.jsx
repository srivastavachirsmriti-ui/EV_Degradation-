function formatNumber(value, digits = 0) {
  if (value === null || value === undefined) return "—";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export default function KpiCards({ kpis, dataset }) {
  const cards = [
    {
      label: "Complaints",
      value: formatNumber(kpis.total_complaints),
      foot: `${formatNumber(dataset?.rows ?? kpis.total_complaints)} records`,
      color: "blue",
    },
    {
      label: "Avg battery health",
      value: `${formatNumber(kpis.avg_battery_health, 1)}%`,
      foot: `across ${formatNumber(kpis.total_complaints)} records`,
      color: "teal",
    },
    {
      label: "Issue categories",
      value: formatNumber(kpis.issue_categories),
      foot: "types of reported concerns",
      color: "graphite",
    },
    {
      label: "High severity",
      value: formatNumber(kpis.high_severity),
      foot: `${(kpis.high_severity_share * 100).toFixed(1)}% of complaints`,
      color: "orange",
    },
  ];

  return (
    <section className="grid kpis">
      {cards.map((card) => (
        <div className="card" key={card.label}>
          <div className="kpi-label">{card.label}</div>
          <div className={`kpi-value ${card.color}`}>{card.value}</div>
          <div className="kpi-foot">{card.foot}</div>
        </div>
      ))}
    </section>
  );
}

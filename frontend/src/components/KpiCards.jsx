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
    },
    {
      label: "Avg battery health",
      value: `${formatNumber(kpis.avg_battery_health, 1)}%`,
      foot: `across ${formatNumber(kpis.total_complaints)} records`,
    },
    {
      label: "Avg degradation",
      value: `${formatNumber(kpis.avg_degradation, 1)}%`,
      foot: `avg age ${formatNumber(kpis.avg_vehicle_age_months, 1)} months`,
    },
    {
      label: "Issue categories",
      value: formatNumber(kpis.issue_categories),
      foot: `${formatNumber(kpis.clusters)} K-Means clusters`,
    },
    {
      label: "High severity",
      value: formatNumber(kpis.high_severity),
      foot: `${(kpis.high_severity_share * 100).toFixed(1)}% of complaints`,
    },
    {
      label: "Avg charge cycles",
      value: formatNumber(kpis.avg_charging_cycles),
      foot: "per vehicle",
    },
  ];

  return (
    <section className="grid kpis">
      {cards.map((card) => (
        <div className="card" key={card.label}>
          <div className="kpi-label">{card.label}</div>
          <div className="kpi-value">{card.value}</div>
          <div className="kpi-foot">{card.foot}</div>
        </div>
      ))}
    </section>
  );
}

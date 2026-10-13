import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function AgeTrendChart({ data, note }) {
  const chartData = data.map((row) => ({
    age: `${row.age_bucket}m`,
    health: row.avg_battery_health,
    degradation: row.avg_degradation,
  }));

  return (
    <div className="card chart-full">
      <h3>Battery health vs. vehicle age</h3>
      <p className="subtitle">{note}</p>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="age" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} unit="%" />
          <Tooltip />
          <Legend />
          <Area
            type="monotone"
            dataKey="health"
            name="Avg battery health"
            stroke="#16a34a"
            fill="#bbf7d0"
            strokeWidth={2}
          />
          <Area
            type="monotone"
            dataKey="degradation"
            name="Avg degradation"
            stroke="#dc2626"
            fill="#fecaca"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

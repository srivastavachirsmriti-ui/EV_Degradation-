import {
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

const COLORS = { High: "#dc2626", Medium: "#f59e0b", Low: "#16a34a" };

export default function SeverityChart({ data }) {
  return (
    <div className="card">
      <h3>Complaint severity</h3>
      <p className="subtitle">Reported severity mix across all complaints</p>
      <ResponsiveContainer width="100%" height={260}>
        <PieChart>
          <Pie
            data={data}
            dataKey="count"
            nameKey="severity"
            innerRadius={55}
            outerRadius={95}
            paddingAngle={2}
          >
            {data.map((entry) => (
              <Cell key={entry.severity} fill={COLORS[entry.severity] || "#94a3b8"} />
            ))}
          </Pie>
          <Tooltip />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

import {
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

const COLORS = { High: "#DA937D", Medium: "#E8BE7E", Low: "#63B4A7" };

export default function SeverityChart({ data }) {
  return (
    <div className="card">
      <h3>How serious are the concerns?</h3>
      <p className="subtitle">Severity labels in the synthetic collection.</p>
      <ResponsiveContainer width="100%" height={268}>
        <PieChart>
          <Pie
            isAnimationActive={false}
            data={data}
            dataKey="count"
            nameKey="severity"
            innerRadius={68}
            outerRadius={94}
            paddingAngle={3}
          >
            {data.map((entry) => (
              <Cell key={entry.severity} fill={COLORS[entry.severity] || "#94a3b8"} />
            ))}
          </Pie>
          <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e8edf3", fontSize: 12 }} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

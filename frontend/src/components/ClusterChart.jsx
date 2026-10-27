import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const COLORS = ["#17324D", "#36566F", "#147D79", "#64ADA7", "#A6B7BF"];

export default function ClusterChart({ data }) {
  const chartData = data.map((item) => ({
    name: `C${item.cluster_id}`,
    count: item.count,
    dominant: item.dominant_issue
      ? `${item.dominant_issue} (${Math.round(item.dominant_share * 100)}%)`
      : "",
  }));

  return (
    <div className="card">
      <h3>K-Means cluster distribution</h3>
      <p className="subtitle">
        Cluster names derived from each cluster&apos;s dominant issue label
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
          <Tooltip
            formatter={(value, _name, entry) => [
              `${value} (${entry.payload.dominant})`,
              "complaints",
            ]}
          />
          <Bar isAnimationActive={false} dataKey="count" radius={[6, 6, 0, 0]}>
            {chartData.map((_, index) => (
              <Cell key={index} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="kpi-foot" style={{ marginTop: 10 }}>
        {data.map((c) => (
          <div key={c.cluster_id}>
            C{c.cluster_id}: {c.dominant_issue} · {c.count} records
          </div>
        ))}
      </div>
    </div>
  );
}

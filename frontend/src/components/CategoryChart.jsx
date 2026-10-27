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

const COLORS = ["#3C86CB", "#52A3CF", "#57B3B1", "#9CBFC7", "#CBD9E3"];

export default function CategoryChart({ data }) {
  return (
    <div className="card">
      <h3>What owners are reporting</h3>
      <p className="subtitle">Capacity loss is the most common concern in this dataset.</p>
      <ResponsiveContainer width="100%" height={268}>
        <BarChart layout="vertical" data={data} margin={{ top: 2, right: 30, left: 0, bottom: 2 }} barSize={22} accessibilityLayer>
          <CartesianGrid stroke="#eef1f5" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 10, fill: "#8a94a1" }} axisLine={false} tickLine={false} allowDecimals={false} />
          <YAxis type="category" dataKey="category" width={134} tick={{ fontSize: 11, fill: "#657181" }} axisLine={false} tickLine={false} />
          <Tooltip cursor={{ fill: "#f5f8fb" }} contentStyle={{ borderRadius: 12, border: "1px solid #e8edf3", fontSize: 12 }} />
          <Bar isAnimationActive={false} dataKey="count" name="Complaints" radius={[0, 5, 5, 0]} label={{ position: "right", fontSize: 11, fill: "#657181" }}>
            {data.map((_, index) => (
              <Cell key={index} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

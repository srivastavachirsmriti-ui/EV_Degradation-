import { Line, LineChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

// One unit and a full 0-100% axis avoid making small synthetic differences look
// dramatic. Numeric labels and context carry the meaning alongside color.
export default function AgeTrendChart({ data }) {
  return <div className="card">
    <h3>Battery health across vehicle ages</h3>
    <p className="subtitle">Average remaining capacity by vehicle age. This is a comparison of different records, not a timeline.</p>
    <ResponsiveContainer width="100%" height={210}>
      <LineChart data={data} margin={{ top: 15, right: 30, left: 0, bottom: 5 }} accessibilityLayer>
        <CartesianGrid stroke="#eef1f5" vertical={false} />
        <XAxis dataKey="age_bucket" tick={{ fontSize: 11, fill: "#84909d" }} tickLine={false} axisLine={false} tickFormatter={(v) => `${v} months`} />
        <YAxis domain={[0,100]} unit="%" tick={{ fontSize: 10, fill: "#84909d" }} tickLine={false} axisLine={false} />
        <Tooltip formatter={(v) => [`${v}%`, "Average battery health"]} labelFormatter={(v) => `Vehicle age: ${v} months`} contentStyle={{ borderRadius: 12, border: "1px solid #e8edf3", fontSize: 12 }} />
        <Line isAnimationActive={false} type="monotone" dataKey="avg_battery_health" name="Average battery health" stroke="#469BBA" strokeWidth={3} dot={{ fill: "white", stroke: "#469BBA", strokeWidth: 2, r: 4 }} activeDot={{ r: 6 }} />
      </LineChart>
    </ResponsiveContainer>
  </div>;
}

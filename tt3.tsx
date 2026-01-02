import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '../ui/chart';

interface ChartSectionProps {
  executionStatsData: Array<{ name: string; value: number; fill: string }>;
}

export default function DashboardCharts({ executionStatsData }: ChartSectionProps) {
  return (
    <ChartContainer
      config={{
        pass: { label: "Passed", color: "#10b981" },
        fail: { label: "Failed", color: "#ef4444" },
        pending: { label: "Pending", color: "#f59e0b" }
      }}
    >
      <ResponsiveContainer width="100%" height={250}>
        <PieChart>
          <Pie
            data={executionStatsData}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={80}
            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
          >
            {executionStatsData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Pie>
          <ChartTooltip content={<ChartTooltipContent />} />
          <Legend wrapperStyle={{ fontSize: '12px' }} />
        </PieChart>
      </ResponsiveContainer>
    </ChartContainer>
  );
}

import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import useFetch from '../hooks/useFetch';
import { Gate } from '../components/Status';

export default function Features() {
  const { data, error } = useFetch(['/features']);
  const gate = Gate({ data, error, text: 'Loading features...' });
  if (gate) return gate;

  const [items] = data;

  return (
    <>
      <h1 className="page-title">Feature Importance</h1>
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">Top 15 Predictive Features</div>
        <ResponsiveContainer width="100%" height={520}>
          <BarChart data={items} layout="vertical" margin={{ top: 5, right: 30, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
            <XAxis type="number" unit="%" tick={{ fill: 'var(--text-dim)', fontSize: 11 }} />
            <YAxis dataKey="label" type="category" tick={{ fill: 'var(--text)', fontSize: 12 }} width={250} interval={0} />
            <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 13 }}
              formatter={(v) => [`${Number(v).toFixed(2)}%`, 'Share of model weight']} />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
              {items.map((_, i) => <Cell key={i} fill={i < 5 ? 'var(--accent)' : 'var(--accent-light)'} fillOpacity={1 - i * 0.04} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="hint">
          Importance is the share of the total absolute weight of the standardised features. Months: 6 = June,
          7 = July, 8 = August. The model uses {'>'}100 features and many of them are related, so no single feature
          has a large share.
        </p>
      </div>

      <div className="card">
        <div className="card-title">Feature Rankings</div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>#</th><th>Feature</th><th>Column</th><th>Share</th><th>Effect</th></tr></thead>
            <tbody>
              {items.map((f, i) => (
                <tr key={f.feature}>
                  <td>{i + 1}</td>
                  <td>{f.label}</td>
                  <td><code>{f.feature}</code></td>
                  <td style={{ fontWeight: 600 }}>{f.importance.toFixed(2)}%</td>
                  <td style={{ color: f.influence === 'raises risk' ? 'var(--red)' : 'var(--green)' }}>
                    {f.influence === 'raises risk' ? 'Higher value raises risk' : 'Higher value lowers risk'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

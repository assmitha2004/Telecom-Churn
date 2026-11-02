import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import useFetch from '../hooks/useFetch';
import { Gate } from '../components/Status';

const NAMES = {
  accuracy: 'Accuracy',
  sensitivity: 'Sensitivity (recall)',
  specificity: 'Specificity',
  precision: 'Precision',
  f1: 'F1 score',
};

export default function ModelPerformance() {
  const { data, error } = useFetch(['/metrics', '/roc']);
  const gate = Gate({ data, error, text: 'Loading metrics...' });
  if (gate) return gate;

  const [metrics, rocRaw] = data;
  const roc = rocRaw.map(p => ({ fpr: p.fpr, tpr: p.tpr }));
  const axisTick = { fill: 'var(--text-dim)', fontSize: 11 };

  return (
    <>
      <h1 className="page-title">Model Performance</h1>
      <div className="grid-2">
        <div className="card">
          <div className="card-title">Test Confusion Matrix (real customers)</div>
          <CM m={metrics.test.confusion_matrix} />
        </div>
        <div className="card">
          <div className="card-title">Train Confusion Matrix (after SMOTE)</div>
          <CM m={metrics.train.confusion_matrix} />
        </div>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">ROC Curve (AUC = {metrics.auc.toFixed(4)})</div>
        <ResponsiveContainer width="100%" height={360}>
          <AreaChart data={roc} margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={axisTick}
              label={{ value: 'False Positive Rate', position: 'bottom', fill: 'var(--text-dim)', fontSize: 12 }} />
            <YAxis type="number" domain={[0, 1]} tick={axisTick}
              label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fill: 'var(--text-dim)', fontSize: 12 }} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--text-dim)" strokeDasharray="4 4" />
            <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 13 }}
              formatter={(v) => [Number(v).toFixed(3), 'True positive rate']}
              labelFormatter={(v) => `False positive rate: ${Number(v).toFixed(3)}`} />
            <Area type="monotone" dataKey="tpr" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.15} strokeWidth={2} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
        <p className="hint">The dashed line is a random guess (AUC 0.5). The higher the curve, the better.
          Average precision on the test set: {metrics.average_precision.toFixed(3)}.</p>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-title">Test Metrics</div>
          <MetricsDetail m={metrics.test} />
        </div>
        <div className="card">
          <div className="card-title">Train Metrics</div>
          <MetricsDetail m={metrics.train} />
        </div>
      </div>
    </>
  );
}

function CM({ m }) {
  return (
    <div className="cm-grid">
      <div className="cm-header" />
      <div className="cm-header">Pred No</div>
      <div className="cm-header">Pred Yes</div>
      <div className="cm-header">Actual No</div>
      <div className="cm-cell cm-tn">{m.tn.toLocaleString()}</div>
      <div className="cm-cell cm-fp">{m.fp.toLocaleString()}</div>
      <div className="cm-header">Actual Yes</div>
      <div className="cm-cell cm-fn">{m.fn.toLocaleString()}</div>
      <div className="cm-cell cm-tp">{m.tp.toLocaleString()}</div>
    </div>
  );
}

function MetricsDetail({ m }) {
  return (
    <table>
      <tbody>
        {Object.keys(NAMES).map(k => (
          <tr key={k}>
            <td>{NAMES[k]}</td>
            <td style={{ fontWeight: 600 }}>{(m[k] * 100).toFixed(2)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

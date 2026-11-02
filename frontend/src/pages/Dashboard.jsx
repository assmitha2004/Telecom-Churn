import React from 'react';
import useFetch from '../hooks/useFetch';
import { Gate } from '../components/Status';

export default function Dashboard() {
  const { data, error } = useFetch(['/stats', '/metrics']);
  const gate = Gate({ data, error, text: 'Loading dashboard...' });
  if (gate) return gate;

  const [stats, metrics] = data;
  const pct = v => `${(v * 100).toFixed(1)}%`;

  const cards = [
    { label: 'Total Records', value: stats.total_records.toLocaleString(), cls: 'accent' },
    { label: 'High-Value Customers', value: stats.high_value_customers.toLocaleString(), cls: 'cyan' },
    { label: 'Churn Rate', value: `${stats.churn_rate}%`, cls: 'red' },
    { label: 'Features Used', value: stats.feature_count, cls: 'accent' },
    { label: 'Model AUC', value: metrics.auc.toFixed(4), cls: 'green' },
    { label: 'Churners Caught', value: pct(metrics.test.sensitivity), cls: 'amber' },
  ];

  const steps = [
    ['Select customers', `Keep the top 30% by average recharge in June and July: ${stats.high_value_customers.toLocaleString()} of ${stats.total_records.toLocaleString()} customers.`],
    ['Tag churn', `A customer churned if there were no calls and no mobile data in September. ${stats.churn_count.toLocaleString()} customers (${stats.churn_rate}%) churned.`],
    ['Remove September data', 'All September columns are removed, so the model cannot see the future.'],
    ['Build trend features', 'Flags show if usage in August fell below the June-July average.'],
    ['Balance the classes', `SMOTE adds synthetic churners to the training data only (${stats.train_size.toLocaleString()} to ${stats.train_size_after_smote.toLocaleString()} rows).`],
    ['Train', `Standardise, reduce to ${metrics.n_components} PCA components (${metrics.variance_explained}% of variance), then fit Logistic Regression.`],
  ];

  return (
    <>
      <h1 className="page-title">Dashboard</h1>
      <div className="stat-grid">
        {cards.map(c => (
          <div className="stat-card" key={c.label}>
            <div className="stat-label">{c.label}</div>
            <div className={`stat-value ${c.cls}`}>{c.value}</div>
          </div>
        ))}
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-title">Test Metrics (real customers, {stats.test_size.toLocaleString()})</div>
          <MetricsTable m={metrics.test} />
        </div>
        <div className="card">
          <div className="card-title">Train Metrics (after SMOTE)</div>
          <MetricsTable m={metrics.train} />
        </div>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">How to read these numbers</div>
        <p className="body-text">
          The model finds <strong>{pct(metrics.test.sensitivity)}</strong> of the customers who really leave.
          But only <strong>{pct(metrics.test.precision)}</strong> of the customers it flags will really leave, because
          only {stats.churn_rate}% of customers churn. So the best use is a shortlist for the retention team,
          not a final decision. Test metrics are the honest ones. Train metrics use balanced, partly synthetic data.
        </p>
      </div>

      <div className="card">
        <div className="card-title">Pipeline</div>
        <ol className="steps">
          {steps.map(([title, text]) => (
            <li key={title}><strong>{title}.</strong> {text}</li>
          ))}
        </ol>
      </div>
    </>
  );
}

function MetricsTable({ m }) {
  const rows = [
    ['Accuracy', m.accuracy], ['Sensitivity (recall)', m.sensitivity],
    ['Specificity', m.specificity], ['Precision', m.precision], ['F1 Score', m.f1],
  ];
  return (
    <table>
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}><td>{k}</td><td style={{ fontWeight: 600 }}>{(v * 100).toFixed(2)}%</td></tr>
        ))}
      </tbody>
    </table>
  );
}

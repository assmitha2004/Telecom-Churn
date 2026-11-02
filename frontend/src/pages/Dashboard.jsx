import React from 'react';
import useFetch from '../hooks/useFetch';
import { Gate } from '../components/Status';

export default function Dashboard() {
  const { data, error } = useFetch(['/stats', '/metrics']);
  const gate = Gate({ data, error, text: 'Loading dashboard...' });
  if (gate) return gate;

  const [stats, metrics] = data;
  const pct = v => `${(v * 100).toFixed(2)}%`;

  const cards = [
    { label: 'Raw Dataset Records', value: stats.total_records.toLocaleString(), cls: 'accent' },
    { label: 'High-Value Cohort (Top 30%)', value: stats.high_value_customers.toLocaleString(), cls: 'cyan' },
    { label: 'Churn Rate in High-Value Cohort', value: `${stats.churn_rate}%`, cls: 'red' },
    { label: 'Model Input Features', value: stats.feature_count.toLocaleString(), cls: 'accent' },
    { label: 'Test ROC-AUC', value: metrics.auc.toFixed(4), cls: 'green' },
    { label: 'Test Recall (Churners Caught)', value: pct(metrics.test.sensitivity), cls: 'amber' },
  ];

  const steps = [
    ['Start with the raw dataset', `${stats.total_records.toLocaleString()} customer records and ${stats.total_columns} original columns.`],
    ['Select the high-value cohort', `Keep the top 30% by average recharge in June and July: ${stats.high_value_customers.toLocaleString()} customers.`],
    ['Define churn', `A customer is labelled churned if there are no calls and no mobile data in September: ${stats.churn_count.toLocaleString()} churners (${stats.churn_rate}% of this cohort).`],
    ['Prevent data leakage', 'September usage columns are removed before training; the model uses information available up to August.'],
    ['Engineer usage trends', 'Flags identify whether August usage fell below the June–July average.'],
    ['Balance training data only', `SMOTE expands the training split from ${stats.train_size.toLocaleString()} to ${stats.train_size_after_smote.toLocaleString()} rows. The test split remains real and untouched.`],
    ['Train and evaluate', `StandardScaler, PCA with ${metrics.n_components} components (${metrics.variance_explained}% variance explained), then Logistic Regression. Evaluation uses ${stats.test_size.toLocaleString()} held-out customers.`],
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

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">Held-Out Test Metrics — Real Customers ({stats.test_size.toLocaleString()})</div>
        <MetricsTable m={metrics.test} />
        <p className="hint" style={{ marginTop: 12 }}>
          These are the primary evaluation metrics. The test set was not oversampled. Precision is 31.13%, meaning
          roughly 31 out of every 100 customers flagged by this model actually churned in this test cohort.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">How to interpret the dashboard</div>
        <p className="body-text">
          The raw dataset contains {stats.total_records.toLocaleString()} records. The model is trained only on the
          high-value cohort of {stats.high_value_customers.toLocaleString()} customers, not on all raw records.
          Therefore, the {stats.churn_rate}% churn rate applies to that selected cohort, not to the full dataset.
          Test recall shows the share of actual churners detected; test precision shows how many flagged customers
          really churned. Use predictions to prioritise retention outreach, not as a final decision.
        </p>
      </div>

      <div className="card">
        <div className="card-title">Model Pipeline</div>
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

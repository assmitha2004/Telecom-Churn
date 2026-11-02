import React, { useState } from 'react';
import api from '../services/api';
import useFetch from '../hooks/useFetch';
import { Gate } from '../components/Status';

const HIDDEN_INPUTS = new Set(['spl_ic_mou_8']);
const INTEGER_INPUTS = new Set([
  'monthly_2g_8', 'monthly_3g_8', 'monthly_2g_6',
  'sachet_2g_8', 'total_rech_num_8', 'monthly_2g_7'
]);
const HELP = {
  monthly_2g_8: 'Number of monthly 2G packs purchased in August.',
  monthly_3g_8: 'Number of monthly 3G packs purchased in August.',
  total_ic_mou_8: 'Total incoming call time during August.',
  monthly_2g_6: 'Number of monthly 2G packs purchased in June.',
  sachet_2g_8: 'Number of short-validity 2G packs purchased in August.',
  loc_ic_t2m_mou_8: 'Incoming local-call minutes from other mobile networks.',
  total_rech_num_8: 'Number of separate recharge transactions in August.',
  loc_og_t2m_mou_8: 'Outgoing local-call minutes to other mobile networks.',
  loc_og_mou_8: 'Total local outgoing call time during August.',
  monthly_2g_7: 'Number of monthly 2G packs purchased in July.',
  arpu_8: 'Average revenue generated per customer in August.'
};
const emptyValues = features => Object.fromEntries(features.map(f => [f.name, '']));

export default function Predict() {
  const { data, error } = useFetch(['/schema']);
  const gate = Gate({ data, error, text: 'Loading form...' });
  if (gate) return gate;
  return <PredictForm schema={data[0]} />;
}

function PredictForm({ schema }) {
  const features = schema.key_features.filter(f => !HIDDEN_INPUTS.has(f.name));
  const [values, setValues] = useState(() => emptyValues(features));
  const [activePreset, setActivePreset] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [problem, setProblem] = useState(null);

  const validate = vals => {
    for (const f of features) {
      const raw = vals[f.name];
      if (raw === '' || raw === null || raw === undefined) continue;
      const n = Number(raw);
      if (!Number.isFinite(n)) return `${f.label}: enter a valid number.`;
      if (n < 0) return `${f.label}: negative values are not allowed.`;
      if (n < f.min || n > f.max) return `${f.label}: enter a value between ${f.min} and ${f.max}.`;
      if (INTEGER_INPUTS.has(f.name) && !Number.isInteger(n)) return `${f.label}: enter a whole number.`;
      if (!INTEGER_INPUTS.has(f.name) && Math.round(n * 100) !== n * 100) return `${f.label}: use up to 2 decimal places.`;
    }
    return null;
  };

  const run = async vals => {
    // Normalize decimal inputs before validation and prediction. Model-derived
    // feature ranges can contain more precision than the form accepts.
    const normalized = Object.fromEntries(Object.entries(vals).map(([key, value]) => {
      if (value === '' || value === null || value === undefined) return [key, value];
      const number = Number(value);
      if (!Number.isFinite(number) || INTEGER_INPUTS.has(key)) return [key, value];
      return [key, Number(number.toFixed(2)).toString()];
    }));
    setValues(normalized);

    const issue = validate(normalized);
    if (issue) { setResult(null); setProblem(issue); return; }
    const payload = {};
    for (const [k, v] of Object.entries(normalized)) if (v !== '' && v !== null) payload[k] = Number(v);
    setLoading(true);
    setProblem(null);
    try {
      const { data } = await api.post('/predict', payload);
      setResult(data);
    } catch (err) {
      setResult(null);
      setProblem(err.response?.data?.error || 'The prediction failed. Please try again.');
    }
    setLoading(false);
  };
  const handleSubmit = e => { e.preventDefault(); setActivePreset(null); run(values); };
  const loadPreset = preset => {
    const filled = Object.fromEntries(features.map(f => [f.name, String(preset.values[f.name] ?? '')]));
    setValues(filled);
    setActivePreset(preset.id);
    run(filled);
  };
  const clear = () => {
    setValues(emptyValues(features));
    setActivePreset(null);
    setResult(null);
    setProblem(null);
  };
  const groups = [
    { title: 'Recharge & Data Packs', names: ['monthly_2g_8','monthly_3g_8','monthly_2g_6','sachet_2g_8','monthly_2g_7'] },
    { title: 'Incoming Calls', names: ['total_ic_mou_8','loc_ic_t2m_mou_8'] },
    { title: 'Outgoing Calls', names: ['loc_og_t2m_mou_8','loc_og_mou_8'] },
    { title: 'Recharge & Spending', names: ['total_rech_num_8','arpu_8'] }
  ];
  return (
    <>
      <h1 className="page-title">Customer Churn Prediction</h1>
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">Try a sample customer</div>
        <div className="chips">
          {schema.presets.map(p => (
            <button key={p.id} type="button" className={`chip ${activePreset === p.id ? 'on' : ''}`}
              onClick={() => loadPreset(p)} title={p.description}>{p.label}</button>
          ))}
        </div>
        <p className="hint">Choose an example to fill the form with sample customer data. You can edit it before predicting.</p>
      </div>
      <div className="card">
        <div className="card-title">Customer Usage Details</div>
        <p className="body-text" style={{ marginBottom: 16 }}>
          Enter information from the customer's telecom usage or billing records for June, July and August.
          Leave a field blank to use the model's typical customer value. Other model inputs are automatically set to typical values.
        </p>
        <form onSubmit={handleSubmit} noValidate>
          {groups.map(group => (
            <section key={group.title} style={{ marginBottom: 22 }}>
              <h3 style={{ margin: '0 0 10px' }}>{group.title}</h3>
              <div className="form-grid">
                {group.names.map(name => {
                  const f = features.find(item => item.name === name);
                  if (!f) return null;
                  const whole = INTEGER_INPUTS.has(name);
                  const max = whole ? Math.floor(f.max) : f.max;
                  return (
                    <div className="field" key={name}>
                      <label htmlFor={name}>{f.label}</label>
                      <input id={name} type="number" min="0" max={max}
                        step={whole ? '1' : '0.01'}
                        placeholder={whole ? 'Enter whole number' : 'Enter value'}
                        value={values[name]}
                        onChange={e => { setValues(v => ({ ...v, [name]: e.target.value })); setProblem(null); }} />
                      <div className="range">{HELP[name]} {whole ? 'Enter a whole number.' : 'Use minutes or rupees as applicable. Extra decimal places are rounded to 2 when predicting.'}</div>
                      <div className="range">Allowed range: 0 to {max}</div>
                    </div>
                  );
                })}
              </div>
            </section>
          ))}
          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading ? 'Predicting...' : 'Predict Customer Churn'}
          </button>
          <button className="btn btn-secondary" type="button" style={{ marginLeft: 8 }} onClick={clear}>Clear All</button>
        </form>
        {problem && <div className="error-box" role="alert" style={{ marginTop: 20 }}>{problem}</div>}
        {result && <Result result={result} />}
      </div>
    </>
  );
}

function Result({ result }) {
  const isChurn = result.prediction === 'Churn';
  const score = Math.round(result.score * 100);
  const baseline = Math.round(result.baseline_score * 100);
  const level = result.risk_level.toLowerCase();
  const maxEffect = Math.max(...result.factors.map(f => Math.abs(f.effect)), 0.0001);

  return (
    <div className={`result-box ${isChurn ? 'churn' : 'safe'}`} aria-live="polite">
      <div className="result-label">Churn risk score</div>
      <div className="result-value" style={{ color: isChurn ? 'var(--red)' : 'var(--green)' }}>
        {score}<span className="out-of"> / 100</span>
      </div>
      <span className={`result-tag ${isChurn ? 'churn' : 'safe'}`}>
        {isChurn ? 'Likely to Churn' : 'Likely to Stay'}
      </span>
      <span className={`level level-${level}`}>{result.risk_level} risk</span>

      <div className="gauge" aria-hidden="true">
        <div className={`gauge-fill ${isChurn ? 'churn' : 'safe'}`} style={{ width: `${score}%` }} />
        <div className="gauge-mark" style={{ left: `${result.threshold * 100}%` }} title="Decision threshold" />
        <div className="gauge-base" style={{ left: `${baseline}%` }} title="Typical customer" />
      </div>
      <div className="gauge-legend">
        <span>| decision threshold ({result.threshold * 100})</span>
        <span>▲ typical customer ({baseline})</span>
      </div>

      {result.factors.length > 0 ? (
        <div className="factors">
          <div className="factors-title">What moved the score (compared with a typical customer)</div>
          {result.factors.map(f => {
            const up = f.effect > 0;
            return (
              <div className="factor" key={f.feature}>
                <div className="factor-name">{f.label}: <strong>{f.value}</strong> <span className="dim">(typical {f.typical})</span></div>
                <div className="factor-bar">
                  <div className={`factor-fill ${up ? 'up' : 'down'}`} style={{ width: `${(Math.abs(f.effect) / maxEffect) * 100}%` }} />
                </div>
                <div className={`factor-dir ${up ? 'up' : 'down'}`}>{up ? 'raises risk' : 'lowers risk'}</div>
              </div>
            );
          })}
        </div>
      ) : (
        <p className="hint" style={{ textAlign: 'center' }}>All values are typical, so this is the score of a typical customer.</p>
      )}

      <p className="hint" style={{ textAlign: 'center', marginTop: 16 }}>
        This is a model score, not a real chance. The model was trained on balanced data, so scores are higher than the real
        churn rate of about 8.6%. Use it to compare customers.
      </p>
    </div>
  );
}

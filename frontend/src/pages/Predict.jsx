import React, { useState } from 'react';
import api from '../services/api';
import useFetch from '../hooks/useFetch';
import { Gate } from '../components/Status';

const emptyValues = features => Object.fromEntries(features.map(f => [f.name, '']));

export default function Predict() {
  const { data, error } = useFetch(['/schema']);
  const gate = Gate({ data, error, text: 'Loading form...' });
  if (gate) return gate;
  return <PredictForm schema={data[0]} />;
}

function PredictForm({ schema }) {
  const features = schema.key_features;
  const [values, setValues] = useState(() => emptyValues(features));
  const [activePreset, setActivePreset] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [problem, setProblem] = useState(null);

  // Send only the fields the user filled in. Empty fields use the typical value on the server.
  const run = async (vals) => {
    const payload = {};
    for (const [k, v] of Object.entries(vals)) if (v !== '' && v !== null) payload[k] = Number(v);
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

  const handleSubmit = (e) => { e.preventDefault(); setActivePreset(null); run(values); };

  const loadPreset = (preset) => {
    const filled = Object.fromEntries(features.map(f => [f.name, String(preset.values[f.name] ?? '')]));
    setValues(filled);
    setActivePreset(preset.id);
    run(filled);
  };

  const clear = () => { setValues(emptyValues(features)); setActivePreset(null); setResult(null); setProblem(null); };

  return (
    <>
      <h1 className="page-title">Predict Churn</h1>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">Start with an example</div>
        <div className="chips">
          {schema.presets.map(p => (
            <button key={p.id} type="button" className={`chip ${activePreset === p.id ? 'on' : ''}`}
              onClick={() => loadPreset(p)} title={p.description}>
              {p.label}
            </button>
          ))}
        </div>
        <p className="hint">Examples use real median values from the data. Click one, then change any field.</p>
      </div>

      <div className="card">
        <div className="card-title">Customer Features</div>
        <p className="body-text" style={{ marginBottom: 16 }}>
          These are the {features.length} features chosen by the model as most useful (months: Jun, Jul, Aug).
          Leave a field empty to use the typical customer value. The other {schema.total_features - features.length} features
          always use typical values.
        </p>
        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            {features.map(f => (
              <div className="field" key={f.name}>
                <label htmlFor={f.name}>{f.label}</label>
                {f.binary ? (
                  <select id={f.name} value={values[f.name]}
                    onChange={e => setValues(v => ({ ...v, [f.name]: e.target.value }))}>
                    <option value="">Typical ({f.default})</option>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                ) : (
                  <input id={f.name} type="number" step="any" placeholder={`Typical: ${f.default}`}
                    value={values[f.name]}
                    onChange={e => setValues(v => ({ ...v, [f.name]: e.target.value }))} />
                )}
                {!f.binary && <div className="range">Usual range: {f.min} to {f.max}</div>}
              </div>
            ))}
          </div>
          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading ? 'Predicting...' : 'Run Prediction'}
          </button>
          <button className="btn btn-secondary" type="button" style={{ marginLeft: 8 }} onClick={clear}>Clear</button>
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

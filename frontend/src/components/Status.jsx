import React from 'react';

export function Loading({ text = 'Loading...' }) {
  return <div className="loading" role="status">{text}</div>;
}

export function ErrorBox({ message }) {
  return <div className="error-box" role="alert">{message}</div>;
}

/** Shows loading or error text, or returns null when the data is ready. */
export function Gate({ data, error, text }) {
  if (error) return <ErrorBox message={error} />;
  if (!data) return <Loading text={text} />;
  return null;
}

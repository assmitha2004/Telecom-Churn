import { useEffect, useState } from 'react';
import api from '../services/api';

/**
 * Load one or more API paths when the page opens.
 * Returns { data, error }. `data` is an array with one result per path.
 */
export default function useFetch(paths) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all(paths.map(p => api.get(p).then(r => r.data)))
      .then(result => { if (!cancelled) setData(result); })
      .catch(() => {
        if (!cancelled) setError('Could not load data from the API. Please check that the backend is running and try again.');
      });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { data, error };
}

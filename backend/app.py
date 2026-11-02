"""
Flask API for the telecom churn project.

Local run:      python backend/app.py        (http://localhost:5000)
Vercel:         api/index.py imports `app` from this file.

All endpoints return JSON. The model itself lives in predictor.py.
"""

from flask import Flask, jsonify, request

from predictor import PredictionError, get_model

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # a prediction request is tiny; refuse big bodies
app.json.sort_keys = False


@app.get("/api/health")
def health():
    get_model()  # fails loudly if model.json is missing
    return jsonify({"status": "ok"})


@app.get("/api/stats")
def stats():
    return jsonify(get_model().stats)


@app.get("/api/metrics")
def metrics():
    return jsonify(get_model().metrics)


@app.get("/api/features")
def features():
    return jsonify(get_model().importances)


@app.get("/api/roc")
def roc():
    return jsonify(get_model().roc)


@app.get("/api/feature-cols")
def feature_cols():
    return jsonify(get_model().feature_cols)


@app.get("/api/schema")
def schema():
    """Inputs for the Predict page: names, labels, typical values, ranges and examples."""
    return jsonify(get_model().schema())


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Send a JSON body."}), 400
    try:
        return jsonify(get_model().predict(data))
    except PredictionError as e:
        return jsonify({"error": str(e)}), 400


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Request is too large."}), 413


@app.errorhandler(FileNotFoundError)
def missing_model(e):
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

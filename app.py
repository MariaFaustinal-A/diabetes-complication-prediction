"""Flask web app: python app.py  ->  http://127.0.0.1:5000"""
import json

import joblib
import numpy as np
import pandas as pd
import csv
import io

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, url_for

import db
from insights import advice, charts, gauge
from ml.config import DATASETS, MODELS, REPORTS
from ml.pipeline import run

app = Flask(__name__)
db.init()
_models = {}


def get_model(key):
    if key not in _models:
        try:
            _models[key] = joblib.load(MODELS / f"{key}.joblib")
        except Exception:                       # missing file or scikit-learn version mismatch -> retrain
            run(key)
            _models[key] = joblib.load(MODELS / f"{key}.joblib")
    return _models[key]


def get_report(key):
    get_model(key)
    return json.loads((REPORTS / f"{key}.json").read_text())


def parse(key, form):
    """Validate raw input -> (one-row DataFrame, errors)."""
    row, errors = {}, []
    for f in DATASETS[key]["fields"]:
        raw = str(form.get(f["name"], "") or "").strip()
        if raw == "":
            if f["required"]:
                errors.append(f"{f['label']} is required.")
            row[f["name"]] = np.nan
        elif f["kind"] == "num" or f.get("dtype") == "num":
            try:
                v = float(raw)
            except ValueError:
                errors.append(f"{f['label']} must be a number.")
                row[f["name"]] = np.nan
                continue
            lo, hi = f.get("min"), f.get("max")
            if lo is not None and not (lo <= v <= hi):
                errors.append(f"{f['label']} should be between {lo} and {hi}.")
            row[f["name"]] = v
        else:
            if raw not in [o[0] for o in f["options"]]:
                errors.append(f"Invalid choice for {f['label']}.")
            row[f["name"]] = raw
    cols = DATASETS[key]["num_cols"] + DATASETS[key]["cat_cols"]
    return pd.DataFrame([row], columns=cols), errors


def probability(key, df):
    return float(get_model(key).predict_proba(df)[0, 1])


COLORS = {"heart": "#e5484d", "kidney": "#3e8ef7", "eye": "#a259e6"}


def stats(rows):
    n = len(rows)
    return dict(total=n, high=sum(r["band"] == "high" for r in rows),
                avg=round(sum(r["probability"] for r in rows) / n, 1) if n else 0,
                by={k: sum(r["disease"] == k for r in rows) for k in DATASETS}, last=rows[0] if n else None)


def trend(rows):
    """Risk-over-time line chart (one line per disease), oldest -> newest."""
    rows, n = list(reversed(rows)), len(rows)
    if not n:
        return None
    W, H, L, R, T, B = 640, 230, 36, 14, 14, 26
    X = lambda i: L + (W - L - R) * (i / (n - 1) if n > 1 else .5)
    Y = lambda p: T + (H - T - B) * (1 - p / 100)
    lines = {}
    for i, r in enumerate(rows):
        lines.setdefault(r["disease"], []).append(
            (round(X(i), 1), round(Y(r["probability"]), 1), f'#{r["id"]} {r["created_at"]}: {r["probability"]}%'))
    return dict(W=W, H=H, L=L, R=R, grid=[dict(y=round(Y(v), 1), v=v) for v in (0, 30, 60, 100)],
                lines=[dict(key=k, color=COLORS[k], pts=v, poly=" ".join(f"{x},{y}" for x, y, _ in v))
                       for k, v in lines.items()])


@app.route("/")
def index():
    return render_template("index.html", reports={k: get_report(k) for k in DATASETS}, datasets=DATASETS,
                           st=stats(db.all_rows()))


def make_result(key, df, prob, report):
    """Build the complete result shown after the Predict risk button is clicked.

    This function is intentionally kept in app.py so the complete web flow is
    easy to trace during project review:

        Predict risk -> predict_risk() -> make_result() -> result page
    """
    p = float(prob)
    if p < 0.30:
        band, band_label = "low", "Low risk"
    elif p < 0.60:
        band, band_label = "moderate", "Moderate risk"
    else:
        band, band_label = "high", "High risk"

    cfg = DATASETS[key]
    value_rows = charts(key, df, report)
    return dict(
        probability=round(p * 100, 1),
        band=band,
        band_label=band_label,
        label={
            "low": cfg["negative"],
            "moderate": cfg["possible"],
            "high": cfg["positive"],
        }[band],
        gauge=gauge(p * 100),
        charts=value_rows,
        chart_note={
            "heart": "Green band = usual target range (blood pressure target is for people with diabetes).",
            "kidney": "Green band = usual laboratory reference range. Your lab's own ranges may differ slightly.",
            "eye": "Green band = middle 50% of training patients WITHOUT retinopathy. These image measures have no clinical normal range.",
        }[key],
        advice=advice(key, band, value_rows),
    )

@app.route("/predict/<key>", methods=["GET", "POST"])
def predict_risk(key):
    """Handle the Predict risk button and display the prediction result.

    POST flow: form input -> validation -> model probability -> make_result()
    -> save history -> render the result on predict.html.
    """
    if key not in DATASETS:
        abort(404)
    report = get_report(key)
    result, errors, values, rid = None, [], {}, None
    if request.method == "POST":
        values = request.form.to_dict()
        df, errors = parse(key, request.form)
        if not errors:
            p = probability(key, df)
            result = make_result(key, df, p, report)
            inputs = {k: v for k, v in values.items() if not k.startswith("_")}
            rid = db.save(key, values.get("_patient", "").strip()[:80], result["probability"],
                          result["band"], result["label"], inputs)
    return render_template("predict.html", key=key, cfg=DATASETS[key], result=result, errors=errors,
                           values=values, report=report, rid=rid)


@app.route("/api/predict/<key>", methods=["POST"])
def api_predict(key):
    if key not in DATASETS:
        abort(404)
    data = request.get_json(force=True, silent=True) or {}
    df, errors = parse(key, data)
    if errors:
        return jsonify(errors=errors), 400
    r = make_result(key, df, probability(key, df), get_report(key))
    rid = db.save(key, str(data.get("_patient", ""))[:80], r["probability"], r["band"], r["label"],
                  {k: v for k, v in data.items() if not k.startswith("_")})
    return jsonify(id=rid, probability=r["probability"], band=r["band"], band_label=r["band_label"],
                   label=r["label"])


def _filters():
    d = request.args.get("disease")
    return (d if d in DATASETS else None), request.args.get("patient", "").strip()


@app.route("/settings")
def settings():
    return render_template("settings.html", count=len(db.all_rows()))


@app.route("/history")
def history():
    d, pat = _filters()
    rows = db.all_rows(d, pat)
    return render_template("history.html", rows=rows, current=d, patient=pat, datasets=DATASETS,
                           st=stats(rows), chart=trend(rows), colors=COLORS)


@app.route("/history/export.csv")
def history_export():
    d, pat = _filters()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["id", "created_at", "patient", "check", "risk_percent", "band", "result", "inputs_json"])
    for r in db.all_rows(d, pat):
        w.writerow([r["id"], r["created_at"], r["patient"], r["disease"], r["probability"], r["band"],
                    r["label"], r["inputs"]])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=prediction_history.csv"})


@app.route("/history/<int:pid>")
def history_view(pid):
    row = db.get(pid)
    if not row:
        abort(404)
    key = row["disease"]
    df, _ = parse(key, json.loads(row["inputs"]))
    report = get_report(key)
    result = make_result(key, df, row["probability"] / 100, report)
    return render_template("history_view.html", row=row, key=key, cfg=DATASETS[key], result=result, report=report)


@app.route("/history/<int:pid>/delete", methods=["POST"])
def history_delete(pid):
    db.delete(pid)
    return redirect(url_for("history"))


@app.errorhandler(404)
def not_found(_):
    return render_template("404.html"), 404


@app.route("/history/clear", methods=["POST"])
def history_clear():
    db.clear()
    return redirect(url_for("history"))


if __name__ == "__main__":
    app.run(debug=True)

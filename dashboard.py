"""Read-only dashboard for reports produced by llm_eval."""
import json
from pathlib import Path
import sqlite3

import streamlit as st

from llm_eval.core import compare, csv_payload, read_runs

st.set_page_config(page_title="LLM Evaluation Platform", layout="wide")
st.title("LLM Evaluation Platform")
source = st.sidebar.radio("Data source", ["JSON reports", "SQLite"])
if source == "SQLite":
    path = st.sidebar.text_input("Database", "evaluation.db")
    try:
        reports = read_runs(Path(path))
    except sqlite3.Error:
        st.warning("Database unavailable.")
        st.stop()
else:
    uploaded = st.sidebar.file_uploader("Evaluation JSON", accept_multiple_files=True, type="json")
    try:
        reports = ([json.load(f) for f in uploaded] if uploaded else
                   [json.loads(p.read_text()) for p in sorted(Path("reports").glob("*/evaluation.json"))])
    except (ValueError, OSError):
        st.error("Unable to read report.")
        st.stop()
mode = st.sidebar.radio("Mode", ["mock", "live"])
reports = sorted((r for r in reports if r["conditions"]["mode"] == mode),
                 key=lambda r: r["timestamp"])
if not reports:
    st.info("No evaluation reports for this mode.")
    st.stop()
st.dataframe([dict(model=r["model_id"], timestamp=r["timestamp"],
                   gate=r["gate"]["passed"], **r["metrics"]) for r in reports])
import pandas as pd

trend = pd.DataFrame([{"timestamp": r["timestamp"], "model": r["model_id"],
                       "pass_rate": r["metrics"]["pass_rate"],
                       "hallucination_rate": r["metrics"]["hallucination_rate"]} for r in reports])
for metric in ("pass_rate", "hallucination_rate"):
    if trend[metric].notna().any():
        st.line_chart(trend, x="timestamp", y=metric, color="model")
selected = st.selectbox("Candidate run", range(len(reports)), index=len(reports)-1,
                        format_func=lambda i: f'{reports[i]["model_id"]} / {reports[i]["timestamp"]}')
report = reports[selected]
baseline = st.selectbox("Baseline run", range(len(reports)),
                        format_func=lambda i: f'{reports[i]["model_id"]} / {reports[i]["timestamp"]}')
try:
    st.subheader("Regression")
    st.json(compare(reports[baseline], report))
except ValueError:
    st.warning("Evaluation conditions differ.")
st.subheader("Confusion matrix")
m = report["confusion"]
st.dataframe(pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                         index=["Actual supported", "Actual unsupported"],
                         columns=["Predicted supported", "Predicted unsupported"]))
st.caption(f'Unclassified: {m["unclassified"]}')
st.subheader("Failed cases")
st.dataframe([row for row in report["cases"] if not row["passed"]])
st.download_button("JSON", json.dumps(report, indent=2), "evaluation.json", "application/json")
st.download_button("CSV", csv_payload(report), "evaluation.csv", "text/csv")

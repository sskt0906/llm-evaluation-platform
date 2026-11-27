"""Read-only dashboard for reports produced by llm_eval."""
import json
from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

from llm_eval.core import compare, csv_payload, outcome_metrics, read_runs


def format_measure(value, suffix="", percent=False):
    if value is None:
        return "未測定"
    return f"{value:.1%}" if percent else f"{value:,}{suffix}"


def display_metrics(report):
    return {**report["metrics"], **outcome_metrics(report["cases"])}


def comparable_conditions(run, selected):
    return (run["conditions"] == selected["conditions"]
            and run["thresholds"] == selected["thresholds"]
            and (run["prompt_version"] != selected["prompt_version"]
                 or run.get("prompt_prefix_sha256") == selected.get("prompt_prefix_sha256")))


def show_dashboard(reports):
    labels = [f'{r["model_id"]} / {r["timestamp"]}' for r in reports]
    selected = st.selectbox("Candidate run", range(len(reports)), index=len(reports) - 1,
                            format_func=labels.__getitem__)
    report = reports[selected]
    metrics = display_metrics(report)
    if report["gate"]["passed"]:
        st.success("Quality gate · PASS")
    else:
        st.error("Quality gate · FAIL · " + ", ".join(report["gate"]["reasons"]))

    cards = [
        ("Answer quality pass rate", "answer_quality_pass_rate"),
        ("Normal case success rate", "normal_case_success_rate"),
        ("Normal system error rate", "normal_system_error_rate"),
        ("Fault injection success rate", "fault_injection_success_rate"),
        ("Legacy pass rate", "pass_rate"),
    ]
    for col, (label, key) in zip(st.columns(len(cards)), cards):
        col.metric(label, format_measure(metrics[key], percent=True))
    normal_count = metrics["normal_case_count"]
    error_count = metrics["normal_system_error_count"]
    st.caption(f"回答品質の評価対象 · {normal_count - error_count} / {normal_count} 通常ケース")
    if error_count:
        st.warning(f"通常ケース {error_count} 件は終端エラーのため回答品質の評価対象外です。")
    if metrics["fault_injection_case_count"]:
        st.caption("障害注入成功率は終端エラーの一致で計算します。再試行回数は品質ゲートで検証します。")

    latency, tokens, cost = st.columns(3)
    latency.metric("P95 latency", format_measure(metrics["p95_latency"], " ms"))
    tokens.metric("Token usage", format_measure(metrics["token_usage"]))
    cost.metric("Cost per successful task", format_measure(metrics["cost_per_successful_task"], " USD"))

    st.subheader("Terminal errors")
    errors = {"timeout": "Timeout", "429": "HTTP 429", "500": "HTTP 500",
              "provider_error": "Provider error", "malformed": "Malformed response"}
    st.dataframe([{"Error": label, "Normal": metrics[f"normal_error_{code}_count"],
                   "Injected": metrics[f"injected_error_{code}_count"]}
                  for code, label in errors.items()], hide_index=True)

    st.subheader("Regression")
    others = [i for i in range(len(reports)) if i != selected]
    if others:
        baseline = st.selectbox("Baseline run", others, format_func=labels.__getitem__)
        try:
            result = compare(reports[baseline], report)
        except ValueError:
            st.warning("Evaluation conditions differ.")
        else:
            if result["regressions"]:
                st.warning("Quality regression · " + ", ".join(result["regressions"]))
            st.json(result)
    else:
        st.info("Regression comparison requires another run.")

    st.subheader("Model quality comparison")
    latest = {}
    history = []
    for run in reports:
        model = run["model_id"]
        values = display_metrics(run)
        if not comparable_conditions(run, report):
            continue
        latest[model] = {
            "Model": model,
            "Answer quality pass rate": values["answer_quality_pass_rate"],
            "Normal case success rate": values["normal_case_success_rate"],
        }
        history.append({"timestamp": run["timestamp"], "model": model,
                        "pass_rate": values["pass_rate"],
                        "hallucination_rate": values["hallucination_rate"]})
    bars = pd.DataFrame(latest.values())
    st.caption("棒グラフと時系列は選択した実行と比較可能な評価条件に絞っています。")
    if bars["Answer quality pass rate"].isna().any():
        st.caption("回答品質が未測定のモデルは回答品質の棒を表示しません。")
    rates = ["Answer quality pass rate", "Normal case success rate"]
    if bars[rates].notna().any().any():
        st.bar_chart(bars, x="Model", y=rates, stack=False)
    else:
        st.info("モデル別の成功率は未測定です。")

    st.subheader("History")
    trend = pd.DataFrame(history)
    trend["timestamp"] = pd.to_datetime(trend["timestamp"], utc=True, format="ISO8601")
    for metric in ("pass_rate", "hallucination_rate"):
        if trend[metric].notna().any():
            st.vega_lite_chart(trend, {
                "mark": {"type": "line", "point": True, "tooltip": True},
                "encoding": {
                    "x": {"field": "timestamp", "type": "temporal"},
                    "y": {"field": metric, "type": "quantitative", "axis": {"format": "%"}},
                    "color": {"field": "model", "type": "nominal"},
                },
            })
    st.dataframe([dict(model=r["model_id"], timestamp=r["timestamp"],
                       evaluator_version=r["conditions"]["evaluator_version"],
                       gate="PASS" if r["gate"]["passed"] else "FAIL",
                       **display_metrics(r)) for r in reports])

    st.subheader("Confusion matrix")
    m = report["confusion"]
    if sum(m.values()):
        st.dataframe(pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                                  index=["Actual supported", "Actual unsupported"],
                                  columns=["Predicted supported", "Predicted unsupported"]))
        st.caption(f'Unclassified · {m["unclassified"]}')
    else:
        st.info("評価器の混同行列は未測定です。")

    st.subheader("Failed cases")
    for row in report["cases"]:
        if not row["passed"]:
            kind = ("Injected fault" if row["expected_error"] else
                    "System error" if row["error"] else "Answer quality")
            with st.expander(f'{row["id"]} · {row["category"]} · {kind}'):
                st.json(row)
    st.download_button("JSON", json.dumps(report, indent=2), "evaluation.json", "application/json")
    st.download_button("CSV", csv_payload(report), "evaluation.csv", "text/csv")


def main():
    st.set_page_config(page_title="LLM Evaluation Platform", layout="wide")
    st.title("LLM Evaluation Platform")
    source = st.sidebar.radio("Data source", ["JSON reports", "SQLite"])
    try:
        if source == "SQLite":
            path = st.sidebar.text_input("Database", "evaluation.db")
            reports = read_runs(Path(path))
        else:
            uploaded = st.sidebar.file_uploader("Evaluation JSON", accept_multiple_files=True, type="json")
            reports = ([json.load(f) for f in uploaded] if uploaded else
                       [json.loads(p.read_text(encoding="utf-8"))
                        for p in sorted(Path("reports").glob("*/evaluation.json"))])
    except sqlite3.Error:
        st.warning("Database unavailable.")
        return
    except (ValueError, OSError):
        st.error("Unable to read report.")
        return
    modes = {"mock": "モック評価", "live": "実LLM評価"}
    mode = st.sidebar.radio("Mode", list(modes), format_func=modes.__getitem__)
    st.caption(modes[mode])
    unique = {}
    conflicts = set()
    for report in reports:
        if report["conditions"]["mode"] != mode:
            continue
        run_id = report["run_id"]
        if run_id in unique and unique[run_id] != report:
            conflicts.add(run_id)
        unique[run_id] = report
    if conflicts:
        st.warning("同じ実行 ID に異なる内容があるため、該当するレポートを除外しました。")
    for run_id in unique.keys() - conflicts:
        report = unique[run_id]
        if any(report["metrics"].get(key, value) != value
               for key, value in outcome_metrics(report["cases"]).items()):
            conflicts.add(run_id)
            st.warning("ケース行と集計値が一致しないレポートを除外しました。")
    reports = sorted((report for run_id, report in unique.items() if run_id not in conflicts),
                     key=lambda r: (r["timestamp"], r["run_id"]))
    if not reports:
        st.info("No evaluation reports for this mode.")
        return
    show_dashboard(reports)


if __name__ == "__main__":
    main()

import csv
import json
from pathlib import Path

import pytest
import pyarrow as pa
from streamlit.testing.v1 import AppTest

from llm_eval.adapters import Failure, Mock
from llm_eval.core import aggregate, outcome_metrics, read_runs, run, save

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("llm_eval.core.sleep", lambda _: None)
    return tmp_path


def evaluate(model="mock-good", adapter=None):
    return run(ROOT / "data/holdout.json", ROOT / "quality.json", model, adapter=adapter)


def app():
    result = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=15).run()
    assert not result.exception
    return result


def cards(view):
    return {metric.label: metric.value for metric in view.metric}


def test_no_data(reports):
    assert "No evaluation reports" in app().info[0].value


def test_single_run_and_storage_agree(reports):
    report = evaluate()
    directory = reports / "reports" / "good"
    db = reports / "evaluation.db"
    save(report, directory, db)
    assert json.loads((directory / "evaluation.json").read_text()) == read_runs(db)[0] == report
    with (directory / "evaluation.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(report["cases"])
    assert {row["run_id"] for row in rows} == {report["run_id"]}
    restored_cases = [
        dict(passed=row["passed"] == "True", error=row["error"] or None,
             expected_error=row["expected_error"] or None)
        for row in rows
    ]
    for field in ("passed", "error", "expected_error"):
        assert [row[field] for row in restored_cases] == [row[field] for row in report["cases"]]
    assert outcome_metrics(restored_cases) == outcome_metrics(report["cases"])
    metrics, _, _ = aggregate(report["cases"])
    assert metrics == report["metrics"]
    assert metrics["pass_rate"] == 6 / 9
    assert metrics["normal_case_success_rate"] == 1
    assert metrics["answer_quality_pass_rate"] == 1
    assert metrics["fault_injection_success_rate"] == 1
    assert metrics["normal_system_error_rate"] == 0
    for error in ("timeout", "429", "500"):
        assert metrics[f"normal_error_{error}_count"] == 0
        assert metrics[f"injected_error_{error}_count"] == 1
    view = app()
    assert any("PASS" in message.value for message in view.success)
    charts = [json.loads(chart.proto.spec) for chart in view.get("vega_lite_chart")]
    assert len(charts) == 3
    assert all(chart["mark"]["point"] for chart in charts[1:])
    assert cards(view)["Normal system error rate"] == "0.0%"
    assert cards(view)["P95 latency"] == "未測定"
    assert any("モック評価" in message.value for message in view.caption)
    sqlite_view = view.sidebar.radio[0].set_value("SQLite").run()
    assert not sqlite_view.exception
    assert cards(sqlite_view) == cards(app())


def test_regression_and_live_mode(reports):
    save(evaluate(), reports / "reports" / "good")
    save(evaluate("mock-bad"), reports / "reports" / "bad")

    cases = json.loads((ROOT / "data/holdout.json").read_text())

    class OfflineAdapter(Mock):
        def generate(self, prompt, timeout):
            if prompt == cases[0]["prompt"]:
                raise Failure("429")
            return super().generate(prompt, timeout)

    live = evaluate("ollama:test-double", OfflineAdapter(cases, False))
    save(live, reports / "reports" / "live")
    view = app()
    assert any("FAIL" in message.value for message in view.error)
    assert any("Quality regression" in message.value for message in view.warning)
    view.sidebar.radio[1].set_value("live").run()
    assert not view.exception
    assert any("実LLM評価" in message.value for message in view.caption)
    assert cards(view)["Fault injection success rate"] == "未測定"
    assert cards(view)["Normal system error rate"] == "16.7%"
    assert cards(view)["Answer quality pass rate"] == "100.0%"


def test_legacy_report(reports):
    report = evaluate()
    report["conditions"]["evaluator_version"] = "2"
    for key in outcome_metrics(report["cases"]):
        del report["metrics"][key]
    save(report, reports / "reports" / "legacy")
    assert cards(app())["Answer quality pass rate"] == "100.0%"


def test_no_answer_and_failed_fault():
    metrics = outcome_metrics([
        {"passed": False, "error": "500", "expected_error": None},
        {"passed": False, "error": "500", "expected_error": "timeout"},
    ])
    assert metrics["normal_case_success_rate"] == 0
    assert metrics["answer_quality_pass_rate"] is None
    assert metrics["normal_system_error_rate"] == 1
    assert metrics["fault_injection_success_rate"] == 0


def test_ignores_incompatible_history_and_duplicate_runs(reports):
    legacy = json.loads((ROOT / "reports/good/evaluation.json").read_text())
    current = evaluate()
    save(legacy, reports / "reports" / "legacy")
    save(current, reports / "reports" / "current")
    save(current, reports / "reports" / "duplicate")
    view = app()
    assert len(view.selectbox[0].options) == 2
    assert any("Evaluation conditions differ" in message.value for message in view.warning)
    history = next(table.value for table in view.dataframe if "timestamp" in table.value)
    assert set(history["evaluator_version"]) == {"2", "3"}

    trend = view.get("vega_lite_chart")[1].proto
    assert pa.ipc.open_stream(trend.data.data).read_all().num_rows == 1


def test_fault_code_match_does_not_override_attempt_gate(reports, monkeypatch):
    from llm_eval import core

    execute = core.execute

    def early_exit(adapter, prompt, config):
        reply, error, attempts, elapsed = execute(adapter, prompt, config)
        return reply, error, 1 if error else attempts, elapsed

    monkeypatch.setattr(core, "execute", early_exit)
    report = evaluate()
    assert report["metrics"]["fault_injection_success_rate"] == 1
    assert report["gate"] == {"passed": False, "reasons": ["fault_handling"]}
    save(report, reports / "reports" / "early")
    view = app()
    assert any("fault_handling" in message.value for message in view.error)
    assert any("終端エラーの一致" in message.value for message in view.caption)


def test_coverage_and_failed_case_labels(reports):
    class MostlyUnavailable:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, timeout):
            self.calls += 1
            if self.calls == 1:
                return Mock(cases, False).generate(prompt, timeout)
            raise Failure("timeout")

    cases = json.loads((ROOT / "data/holdout.json").read_text())
    result = evaluate("ollama:test-double", MostlyUnavailable())
    save(result, reports / "reports" / "mostly-unavailable")
    view = app().sidebar.radio[1].set_value("live").run()
    assert not view.exception
    assert cards(view)["Answer quality pass rate"] == "100.0%"
    assert cards(view)["Normal case success rate"] == "16.7%"
    assert any("1 / 6" in message.value for message in view.caption)
    assert any("5 件" in message.value for message in view.warning)
    assert all("System error" in item.label for item in view.expander)


def test_rejects_conflicting_aggregates(reports):
    report = evaluate()
    report["metrics"]["answer_quality_pass_rate"] = 0.5
    save(report, reports / "reports" / "inconsistent")
    view = app()
    assert any("ケース行と集計値が一致しない" in message.value for message in view.warning)
    assert not view.metric


def test_rejects_cross_version_and_cross_mode_comparisons(reports):
    from llm_eval.core import compare

    legacy = json.loads((ROOT / "reports/good/evaluation.json").read_text())
    legacy_bad = json.loads((ROOT / "reports/bad/evaluation.json").read_text())
    assert compare(legacy, legacy_bad)["regressions"]
    current = evaluate()
    with pytest.raises(ValueError, match="incompatible evaluation conditions"):
        compare(legacy, current)
    cases = json.loads((ROOT / "data/holdout.json").read_text())
    live = evaluate("ollama:test-double", Mock(cases, False))
    with pytest.raises(ValueError, match="incompatible evaluation conditions"):
        compare(current, live)
    assert all(row["expected_error"] is None for row in live["cases"])

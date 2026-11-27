from pathlib import Path
import json
import pytest
from llm_eval.core import run, compare, save, read_runs

ROOT = Path(__file__).resolve().parents[1]

def evaluate(model="mock-good", split="holdout"):
    return run(ROOT / f"data/{split}.json", ROOT / "quality.json", model)

def test_gate_and_regression():
    good, bad = evaluate(), evaluate("mock-bad")
    assert good["gate"]["passed"]
    assert not bad["gate"]["passed"]
    assert bad["metrics"]["pass_rate"] < 1
    assert bad["metrics"]["hallucination_rate"] > 0
    assert bad["metrics"]["safety_violations"] > 0
    assert "accuracy" in compare(good, bad)["regressions"]

def test_incompatible():
    with pytest.raises(ValueError):
        compare(evaluate(), evaluate(split="dev"))

def test_labels_and_failures():
    result = evaluate()
    assert result["confusion"]["fp"] == result["confusion"]["fn"] == 0
    assert result["metrics"]["error_rate"] > 0
    assert result["metrics"]["cost_per_successful_task"] is None

def test_round_trip(tmp_path):
    result = evaluate()
    save(result, tmp_path, tmp_path / "eval.db")
    assert read_runs(tmp_path / "eval.db") == [result]
    assert json.loads((tmp_path / "evaluation.json").read_text()) == result
    assert len((tmp_path / "evaluation.csv").read_text().splitlines()) == len(result["cases"]) + 1


def test_schema_rejects_duplicate_keys_and_boolean():
    from llm_eval.core import check

    rule = {"integer_field": "count", "min": 0, "max": 100}
    assert check('{"count":0}', rule)
    assert check('{"count":100}', rule)
    for text in ('{"count":0,"count":1}', '{"count":true}', '{"count":101}'):
        assert not check(text, rule)


def test_retry_sanitizes_failures_and_stops():
    from llm_eval.adapters import Failure
    from llm_eval.core import execute

    class Broken:
        def generate(self, prompt, timeout):
            raise Failure(prompt)

    config = {"timeout_seconds": 1, "max_attempts": 3, "retry_delay_seconds": 0}
    assert execute(Broken(), "timeout", config)[1:3] == ("timeout", 3)
    assert execute(Broken(), "secret-api-key", config)[1:3] == ("provider_error", 1)


def test_unclassified_positive_is_not_a_success():
    from llm_eval.core import aggregate

    cases = evaluate("mock-bad")["cases"]
    positive = next(row for row in cases if row["human_hallucination"])
    positive["hallucination"] = None
    metrics, confusion, _ = aggregate(cases)
    assert confusion["unclassified"] == 1
    assert metrics["recall"] == 0.5


def test_cost_includes_measured_failed_tasks():
    from llm_eval.core import aggregate

    rows = evaluate()["cases"][:2]
    rows[0].update(cost_usd=0.01)
    rows[1].update(cost_usd=0.02, passed=False)
    assert aggregate(rows)[0]["cost_per_successful_task"] == pytest.approx(0.03)
    rows[1]["cost_usd"] = None
    assert aggregate(rows)[0]["cost_per_successful_task"] is None


def test_csv_protects_spreadsheet_formula():
    import csv
    from llm_eval.core import csv_payload

    report = evaluate()
    report["cases"][0]["id"] = "=1+1"
    assert next(csv.DictReader(csv_payload(report).splitlines()))["id"] == "'=1+1"


def test_safety_is_a_veto(tmp_path):
    quality = json.loads((ROOT / "quality.json").read_text())
    quality["minimum"] = {}
    quality["maximum"] = {}
    path = tmp_path / "permissive-test-policy.json"
    path.write_text(json.dumps(quality))
    result = run(ROOT / "data/holdout.json", path, "mock-bad")
    assert result["gate"] == {"passed": False, "reasons": ["safety_violations"]}


def test_prompt_change_requires_version():
    baseline = evaluate()
    args = (ROOT / "data/holdout.json", ROOT / "quality.json", "mock-good")
    changed = run(*args, prompt_prefix="Answer concisely.")
    with pytest.raises(ValueError):
        compare(baseline, changed)
    assert compare(baseline, changed | {"prompt_version": "2"})["passed"]

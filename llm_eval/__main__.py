import argparse
import json
from pathlib import Path

from .core import aggregate, compare, run, save


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("run")
    evaluate.add_argument("--dataset", type=Path, default=Path("data/holdout.json"))
    evaluate.add_argument("--quality", type=Path, default=Path("quality.json"))
    evaluate.add_argument("--model", default="mock-good")
    evaluate.add_argument("--allow-live", action="store_true")
    evaluate.add_argument("--model-version", default="1")
    evaluate.add_argument("--prompt-version", default="1")
    evaluate.add_argument("--prompt-prefix-file", type=Path)
    evaluate.add_argument("--judge")
    evaluate.add_argument("--prices", type=float, nargs=2)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--db", type=Path)
    for name in ("compare", "detector"):
        command = commands.add_parser(name)
        command.add_argument("baseline", type=Path)
        command.add_argument("candidate", type=Path)
        command.add_argument("--output", type=Path)
        if name == "detector":
            command.add_argument("--quality", type=Path, default=Path("quality.json"))
    args = parser.parse_args()
    if args.command == "run" and (args.model not in {"mock-good", "mock-bad"} or args.judge):
        if not args.allow_live:
            parser.error("live calls require --allow-live")
    try:
        if args.command == "run":
            prefix = args.prompt_prefix_file.read_text(encoding="utf-8") if args.prompt_prefix_file else ""
            result = run(args.dataset, args.quality, args.model, args.model_version,
                         args.prompt_version, args.prices, args.judge, prompt_prefix=prefix)
            save(result, args.output, args.db)
            print(json.dumps(result["gate"]))
            return int(not result["gate"]["passed"])
        baseline, candidate = load(args.baseline), load(args.candidate)
        result = compare(baseline, candidate)
        if args.command == "detector":
            if (baseline["model_id"], candidate["model_id"]) != ("mock-good", "mock-bad"):
                raise ValueError("detector requires good and bad mock reports")
            metrics, confusion, _ = aggregate(baseline["cases"] + candidate["cases"])
            limits = load(args.quality)["detector_minimum"]
            result = {"confusion": confusion, "metrics": {
                k: metrics[k] for k in ("precision", "recall", "f1_score",
                                        "false_positive_rate", "false_negative_rate")},
                "passed": not confusion["unclassified"] and all(
                    metrics[k] is not None and metrics[k] >= v for k, v in limits.items())}
        payload = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return int(not result["passed"])
    except (ValueError, KeyError, OSError):
        parser.exit(2, "Evaluation could not proceed. Check input files and settings.\n")


if __name__ == "__main__":
    raise SystemExit(main())

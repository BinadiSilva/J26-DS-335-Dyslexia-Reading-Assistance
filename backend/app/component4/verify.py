"""Repeatable demo-route and latency verification using a temporary database."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from .analytics import learner_report
from .synthetic import seed_demo
from .web import create_app


def verify(model_dir):
    """Exercise the real saved model without modifying the user's records."""
    if not (Path(model_dir) / "model.joblib").is_file():
        raise FileNotFoundError("Train the demonstration model before verification.")
    with TemporaryDirectory(prefix="component4-check-") as directory:
        started = perf_counter()
        app = create_app(Path(directory) / "check.sqlite3", model_dir, secret_key="temporary-verification-only")
        load_seconds = perf_counter() - started
        app.config["TESTING"] = True
        store = app.extensions["c4_store"]
        model = app.extensions["c4_model"]
        seed_demo(store)
        store.create_user("verification", "temporary-verification-password", "teacher")
        client = app.test_client()
        # Authentication itself is tested in test_standalone_web. Here a test
        # session isolates page and prediction latency from password hashing.
        with client.session_transaction() as session:
            session["username"] = "verification"
        routes = {}
        for route in ("/", "/learners/DEMO0001", "/evaluation", "/evaluation.json",
                      "/learners/DEMO0001/report.json", "/training-sessions.csv?source=synthetic"):
            started = perf_counter()
            response = client.get(route)
            if response.status_code != 200:
                raise AssertionError(f"{route}: HTTP {response.status_code}")
            routes[route] = {"status": response.status_code, "seconds": perf_counter() - started}
        timings, statuses = [], {}
        for _ in range(3):
            for learner in store.learners():
                started = perf_counter()
                report = learner_report(store, learner["id"], model)
                timings.append(perf_counter() - started)
                statuses[learner["id"]] = report["forecast"]["status"]
        return {
            "scope": "Local synthetic demonstration; Flask test client, not a load test or browser rendering benchmark.",
            "model": model.predictor.name, "source": model.source,
            "model_sha256": model.report.get("model_sha256"),
            "startup_model_load_seconds": load_seconds, "routes": routes,
            "report_samples": len(timings), "report_p95_seconds": float(np.percentile(timings, 95)),
            "report_max_seconds": max(timings), "target_seconds": 5,
            "report_target_met": max(timings) < 5, "forecast_statuses": statuses,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        parser.error("Choose a new output path; previous evidence is preserved.")
    report = verify(args.model_dir)
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(result)
    print(result)


if __name__ == "__main__":
    main()

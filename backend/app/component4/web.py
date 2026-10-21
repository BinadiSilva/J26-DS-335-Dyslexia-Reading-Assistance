"""Local standalone dashboard: python -m backend.app.component4.web --demo."""

import argparse
from datetime import timedelta
from functools import wraps
import getpass
import json
import os
from pathlib import Path
import secrets
import time

from flask import Flask, abort, flash, g, jsonify, redirect, render_template, request, session, url_for

from .analytics import learner_report
from .models import load_model
from .schema import TARGETS
from .store import Store
from .synthetic import seed_demo

ROOT = Path(__file__).resolve().parents[3]


def chart(sessions, metric, forecast):
    """SVG coordinates with a real date axis, plus a separate forecast segment."""
    from datetime import date
    if not sessions:
        return None
    dates = [date.fromisoformat(row["assessed_on"]).toordinal() for row in sessions]
    values = [row[metric] for row in sessions]
    has_forecast = forecast.get("status") == "available"
    all_dates = dates + ([date.fromisoformat(forecast["date"]).toordinal()] if has_forecast else [])
    all_values = values + ([forecast["values"][metric]] if has_forecast else [])
    top = max(max(all_values) * 1.15, 10)
    def xy(day, value):
        return f"{35 + 380 * (day - dates[0]) / max(1, all_dates[-1] - dates[0]):.2f},{145 - 120 * value / top:.2f}"
    return {"points": " ".join(xy(d, v) for d, v in zip(dates, values)),
            "forecast": " ".join([xy(dates[-1], values[-1]), xy(all_dates[-1], all_values[-1])]) if has_forecast else None,
            "top": round(top), "start": sessions[0]["assessed_on"],
            "end": forecast["date"] if has_forecast else sessions[-1]["assessed_on"]}


def create_app(db_path=None, model_dir=None, secret_key=None):
    app = Flask(__name__)
    database = Path(db_path or ROOT / "runtime/component4.sqlite3")
    store = Store(database)
    if secret_key is None:
        key_path = database.with_suffix(".secret")
        if not key_path.exists():
            descriptor = os.open(key_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "w") as handle:
                handle.write(secrets.token_hex(32))
        secret_key = key_path.read_text().strip()
    app.config.update(SECRET_KEY=secret_key, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict",
                      PERMANENT_SESSION_LIFETIME=timedelta(hours=2), MAX_CONTENT_LENGTH=65536,
                      TRUSTED_HOSTS=["localhost", "127.0.0.1"])
    model = None
    if model_dir and (Path(model_dir) / "model.joblib").exists():
        model = load_model(model_dir)
    app.extensions.update(c4_store=store, c4_model=model)
    failed_logins = {}

    @app.before_request
    def protect():
        g.user = store.user(session.get("username", ""))
        session.setdefault("csrf", secrets.token_urlsafe(32))
        if request.method == "POST":
            supplied = request.form.get("csrf", "")
            if not secrets.compare_digest(supplied, session["csrf"]):
                abort(400, "The form expired. Reload the page and try again.")
        if request.endpoint not in {"login", "static"} and not g.user:
            return redirect(url_for("login"))

    @app.after_request
    def headers(response):
        response.headers.update({
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", "Referrer-Policy": "same-origin",
            "Content-Security-Policy": "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; form-action 'self'; frame-ancestors 'none'; base-uri 'self'",
        })
        return response

    def teacher_only(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if g.user["role"] != "teacher":
                abort(403)
            return view(*args, **kwargs)
        return wrapped

    def authorized(learner_id):
        if g.user["role"] == "parent" and g.user["learner_id"] != learner_id:
            abort(404)
        try:
            return store.learner(learner_id)
        except LookupError:
            abort(404)

    @app.context_processor
    def context():
        return {"csrf": session.get("csrf"), "user": g.get("user")}

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            key = request.remote_addr
            attempts = [stamp for stamp in failed_logins.get(key, []) if time.monotonic() - stamp < 300]
            if len(attempts) >= 10:
                abort(429, "Too many attempts. Try again in five minutes.")
            account = store.authenticate(request.form.get("username", ""), request.form.get("password", ""))
            if account:
                failed_logins.pop(key, None)
                session.clear()
                session.update(username=account["username"], csrf=secrets.token_urlsafe(32))
                session.permanent = True
                return redirect(url_for("home"))
            failed_logins[key] = attempts + [time.monotonic()]
            flash("The username or password is incorrect.", "error")
        return render_template("login.html")

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.get("/")
    def home():
        if g.user["role"] == "parent":
            return redirect(url_for("learner", learner_id=g.user["learner_id"]))
        query = request.args.get("q", "").strip().lower()
        learners = [item for item in store.learners() if query in item["alias"].lower() or query in item["id"].lower()]
        reports = [learner_report(store, item["id"], model) for item in learners]
        return render_template("home.html", reports=reports, query=query,
                               high=sum(r["risk"]["level"] == "High" for r in reports),
                               active=sum(r["active_intervention"] is not None for r in reports))

    @app.post("/learners")
    @teacher_only
    def add_learner():
        try:
            item = store.create_learner(request.form)
        except ValueError as error:
            flash(str(error), "error")
            return redirect(url_for("home"))
        flash("Learner registered. Add an assessment to begin.", "success")
        return redirect(url_for("learner", learner_id=item["id"]))

    @app.get("/learners/<learner_id>")
    def learner(learner_id):
        authorized(learner_id)
        started = time.perf_counter()
        report = learner_report(store, learner_id, model)
        charts = {key: chart(report["sessions"], key, report["forecast"]) for key in TARGETS[:3]}
        return render_template("learner.html", report=report, charts=charts,
                               report_ms=round((time.perf_counter() - started) * 1000, 1))

    @app.post("/learners/<learner_id>/sessions")
    @teacher_only
    def add_session(learner_id):
        authorized(learner_id)
        try:
            store.add_session(learner_id, request.form)
            flash("Assessment saved. Trends, forecast and proposed support have been refreshed.", "success")
        except ValueError as error:
            flash(str(error), "error")
        return redirect(url_for("learner", learner_id=learner_id))

    @app.post("/learners/<learner_id>/interventions")
    @teacher_only
    def start_intervention(learner_id):
        authorized(learner_id)
        try:
            report = learner_report(store, learner_id, model)
            proposal = report["proposed_plan"] or {}
            approved = {**proposal, **request.form.to_dict(), "approved_by": g.user["username"]}
            approved.pop("csrf", None)
            store.start_intervention(learner_id, approved)
            flash("Reviewed plan saved with its baseline assessment.", "success")
        except ValueError as error:
            flash(str(error), "error")
        return redirect(url_for("learner", learner_id=learner_id))

    @app.post("/learners/<learner_id>/interventions/<int:intervention_id>/complete")
    @teacher_only
    def complete_intervention(learner_id, intervention_id):
        authorized(learner_id)
        try:
            store.complete_intervention(learner_id, intervention_id, request.form.get("outcome_session_id"),
                                        request.form.get("adherence"), request.form.get("outcome_notes", ""))
            flash("Outcome recorded. Future proposals now consider this intervention history.", "success")
        except ValueError as error:
            flash(str(error), "error")
        return redirect(url_for("learner", learner_id=learner_id))

    @app.post("/learners/<learner_id>/notes")
    @teacher_only
    def add_note(learner_id):
        authorized(learner_id)
        try:
            store.add_note(learner_id, g.user["username"], request.form.get("body"))
            flash("Professional note saved.", "success")
        except ValueError as error:
            flash(str(error), "error")
        return redirect(url_for("learner", learner_id=learner_id))

    @app.get("/learners/<learner_id>/report.json")
    def export_report(learner_id):
        authorized(learner_id)
        report = learner_report(store, learner_id, model)
        if g.user["role"] == "parent":
            report.pop("notes")
            for item in report["sessions"]:
                item.pop("notes", None)
        response = jsonify(report)
        response.headers["Content-Disposition"] = f'attachment; filename="component4-{learner_id}.json"'
        return response

    @app.get("/evaluation")
    @teacher_only
    def evaluation():
        return render_template("evaluation.html", evaluation=model.report if model else None)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(ROOT / "runtime/component4.sqlite3"))
    parser.add_argument("--model-dir", default=str(ROOT / "models/component4/session_demo_v1"))
    parser.add_argument("--demo", action="store_true", help="Seed only fictional learners; safe to repeat")
    parser.add_argument("--port", default=5054, type=int)
    parser.add_argument("--add-parent", help="Create a parent login, then exit")
    parser.add_argument("--learner", help="Learner ID assigned to the new parent")
    args = parser.parse_args()
    app = create_app(args.db, args.model_dir)
    store = app.extensions["c4_store"]
    if args.add_parent:
        if not args.learner:
            parser.error("--add-parent requires --learner")
        store.learner(args.learner)
        store.create_user(args.add_parent, getpass.getpass("New parent password (12+ characters): "), "parent", args.learner)
        print("Parent account created.")
        return
    if args.demo:
        seed_demo(store)
    if store.user("teacher") is None:
        password = os.environ.get("C4_TEACHER_PASSWORD") or secrets.token_urlsafe(15)
        store.create_user("teacher", password, "teacher")
        print(f"New local teacher login — username: teacher; password: {password}", flush=True)
        print("Save this password; it will not be printed on subsequent launches.", flush=True)
    print(f"Component 4: http://127.0.0.1:{args.port} (local machine only)", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()

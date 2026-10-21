"""HTTP end-to-end workflow, role boundaries, CSRF and escaped display data."""
import re
import pytest

from backend.app.component4.web import create_app
from backend.app.component4.synthetic import generate_cohort


@pytest.fixture
def app(tmp_path):
    app = create_app(tmp_path / "web.sqlite3", secret_key="test-only-secret")
    app.config.update(TESTING=True)
    store = app.extensions["c4_store"]
    for learner_id in ("one", "two"):
        store.create_learner({"alias": learner_id, "age": 9, "grade": 4, "source": "synthetic",
                             "target_speed": 80, "target_accuracy": 90, "target_fluency": 75}, learner_id)
    store.create_user("teacher", "testing-password", "teacher")
    store.create_user("parent", "testing-password", "parent", "one")
    return app


def login(client, username="teacher"):
    page = client.get("/login")
    token = re.search(rb'name="csrf" value="([^"]+)"', page.data)[1].decode()
    return client.post("/login", data={"csrf": token, "username": username, "password": "testing-password"})


def post(client, path, data):
    with client.session_transaction() as session:
        token = session["csrf"]
    return client.post(path, data={**data, "csrf": token}, follow_redirects=True)


def test_authentication_required_and_forged_posts_rejected(app):
    client = app.test_client()
    assert client.get("/learners/one/report.json").status_code == 302
    assert login(client).status_code == 302
    assert client.post("/learners/one/sessions", data={}).status_code == 400
    assert client.get("/learners/unknown").status_code == 404


def test_parent_cannot_access_another_learner_or_mutate_records(app):
    client = app.test_client()
    login(client, "parent")
    assert client.get("/learners/one").status_code == 200
    assert client.get("/learners/two").status_code == 404
    assert client.get("/learners/two/report.json").status_code == 404
    assert client.get("/evaluation").status_code == 403
    assert post(client, "/learners/one/notes", {"body": "denied"}).status_code == 403
    assert post(client, "/learners", {}).status_code == 403


def test_complete_assessment_plan_followup_and_export_workflow(app):
    client = app.test_client()
    login(client)
    rows = generate_cohort(1, 8).to_dict("records")
    for row in rows[:4]:
        page = post(client, "/learners/one/sessions", row)
        assert page.status_code == 200 and b"Assessment saved" in page.data
    page = post(client, "/learners/one/interventions", {"started_on": rows[3]["assessed_on"],
        "target_skill": "fluency", "activity": "Paired reading", "minutes": 10, "frequency": 3, "review_days": 14})
    assert b"Reviewed plan saved" in page.data
    post(client, "/learners/one/sessions", rows[4])
    store = app.extensions["c4_store"]
    intervention = store.interventions("one")[0]
    outcome = store.sessions("one")[-1]
    page = post(client, f"/learners/one/interventions/{intervention['id']}/complete", {"outcome_session_id": outcome["id"], "adherence": 80})
    assert b"Outcome recorded" in page.data
    exported = client.get("/learners/one/report.json")
    assert len(exported.json["sessions"]) == 5
    assert exported.json["interventions"][0]["status"] == "completed"
    assert exported.headers["Cache-Control"] == "no-store"
    assert "attachment" in exported.headers["Content-Disposition"]
    assert client.get("/").status_code == 200
    assert client.get("/evaluation").status_code == 200


def test_notes_are_escaped_and_not_exposed_to_parent(app):
    teacher = app.test_client()
    login(teacher)
    page = post(teacher, "/learners/one/notes", {"body": "<script>alert('x')</script>"})
    assert b"&lt;script&gt;" in page.data and b"<script>alert" not in page.data
    parent = app.test_client()
    login(parent, "parent")
    assert b"alert('x')" not in parent.get("/learners/one").data
    assert "notes" not in parent.get("/learners/one/report.json").json


def test_duplicate_assessment_is_a_visible_error_and_does_not_replace(app):
    client = app.test_client()
    login(client)
    row = generate_cohort(1, 5).iloc[0].to_dict()
    post(client, "/learners/one/sessions", row)
    page = post(client, "/learners/one/sessions", {**row, "fluency_score": 99})
    assert b"already has an assessment" in page.data
    assert app.extensions["c4_store"].sessions("one")[0]["fluency_score"] == row["fluency_score"]


def test_logout_revokes_access_and_post_is_required(app):
    client = app.test_client()
    login(client)
    assert client.get("/logout").status_code == 405
    post(client, "/logout", {})
    assert client.get("/learners/one").status_code == 302

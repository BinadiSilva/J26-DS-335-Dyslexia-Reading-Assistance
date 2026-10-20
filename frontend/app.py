from pathlib import Path
import csv
import io
import json, os
import requests
import streamlit as st

st.set_page_config(page_title="Lumi-It | Adaptive Reading Assistant", page_icon="📚", layout="wide")
ROOT = Path(__file__).resolve().parents[1]
CONTENT_PATH = ROOT / "datasets" / "c3" / "activity_content.json"
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "lumi_it_logo.jpeg"
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.markdown("""<style>
.hero{padding:1.3rem 1.5rem;border-radius:16px;background:linear-gradient(120deg,#eef5ff,#f5efff);color:#202642;margin-bottom:1rem}
.hero h1{margin-bottom:.25rem}
</style>""", unsafe_allow_html=True)

def api_request(method, endpoint, **kwargs):
    try:
        response = requests.request(method, f"{API_BASE_URL}{endpoint}", timeout=30, **kwargs)
        if response.status_code >= 400:
            try: detail = response.json().get("detail", response.text)
            except ValueError: detail = response.text
            st.error(f"Backend returned {response.status_code}: {detail}")
            return None
        return response.json() if response.content else {}
    except requests.RequestException as exc:
        st.error(f"Could not connect to the backend. Start FastAPI and try again. Details: {exc}")
        return None

def load_content():
    try:
        data = json.loads(CONTENT_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(x.get("item_id", x.get("activity_id", ""))): x for x in data if x.get("item_id", x.get("activity_id"))}
        if isinstance(data, dict):
            if isinstance(data.get("activities"), list):
                return {str(x.get("item_id", x.get("activity_id", ""))): x for x in data["activities"] if x.get("item_id", x.get("activity_id"))}
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}

ACTIVITY_CONTENT = load_content()

def activity_id_of(activity):
    return str(activity.get("item_id") or activity.get("activity_id") or activity.get("id") or "")

def plan_activities(data):
    if isinstance(data, list): return data
    if isinstance(data, dict):
        for key in ("activities", "plan", "weekly_plan", "recommendations"):
            value = data.get(key)
            if isinstance(value, list): return value
            if isinstance(value, dict) and isinstance(value.get("activities"), list): return value["activities"]
    return []

def show_activity(learner_id, activity):
    aid = activity_id_of(activity)
    content = {**ACTIVITY_CONTENT.get(aid, {}), **activity}
    st.markdown(f"#### 📖 {content.get('title', 'Reading activity')}")
    st.write(content.get("instructions") or content.get("content") or "Read carefully and complete the activity.")
    passage = content.get("passage") or content.get("story") or content.get("reading_passage")
    if passage:
        st.markdown("##### Read the story")
        st.info(passage)
    elif content.get("content"):
        st.markdown("##### Read / practise")
        st.write(content["content"])

    questions = content.get("questions", [])
    if not isinstance(questions, list) or not questions:
        st.info("No interactive quiz is configured for this activity yet. You can still record your result.")
        with st.form(f"result_{learner_id}_{aid}"):
            score = st.slider("Activity score (%)", 0, 100, 70, key=f"score_{learner_id}_{aid}")
            completed = st.checkbox("I completed this activity", True, key=f"done_{learner_id}_{aid}")
            save = st.form_submit_button("Save Activity Result", type="primary")
        if save:
            result = api_request("POST", "/progress/record", json={"learner_id": learner_id, "item_id": aid, "score": float(score), "completed": bool(completed)})
            if result is not None: st.success("Activity result saved.")
        return

    with st.form(f"quiz_{learner_id}_{aid}"):
        answers = []
        for i, question in enumerate(questions):
            st.markdown(f"**Question {i+1} of {len(questions)}**")
            st.write(question.get("question", f"Question {i+1}"))
            options = question.get("options", [])
            answers.append(st.radio("Choose your answer", options, index=None, key=f"ans_{learner_id}_{aid}_{i}", label_visibility="collapsed") if options else None)
        submitted = st.form_submit_button("Check Answers", type="primary")
    if submitted:
        if any(answer is None for answer in answers):
            st.warning("Please answer every question.")
            return
        correct = sum(answer == question.get("answer") for answer, question in zip(answers, questions))
        score = round(correct / len(questions) * 100)
        for i, (answer, question) in enumerate(zip(answers, questions)):
            if answer == question.get("answer"):
                st.success(f"Question {i+1}: Correct! {question.get('feedback', 'Well done!')}")
            else:
                st.error(f"Question {i+1}: The correct answer is {question.get('answer', 'not configured')}.")
        st.metric("Your score", f"{score}%")
        st.write(f"You answered {correct} out of {len(questions)} correctly.")
        st.session_state[f"quiz_score_{learner_id}_{aid}"] = score
    result_key = f"quiz_score_{learner_id}_{aid}"
    if result_key in st.session_state and st.button("Save Quiz Result", type="primary", key=f"save_{learner_id}_{aid}"):
        result = api_request("POST", "/progress/record", json={"learner_id": learner_id, "item_id": aid, "score": float(st.session_state[result_key]), "completed": True})
        if result is not None: st.success("Quiz result saved successfully!")

with st.sidebar:
    if LOGO_PATH.exists(): st.image(str(LOGO_PATH), use_container_width=True)
    else: st.markdown("## 📚 Lumi-It")
    st.caption("AI-Based Adaptive Reading Assistance System")
    st.divider()
    st.write("Personalized weekly reading plan")
    st.write("Adaptive activities and progress tracking")
    if st.button("Check Backend Connection", use_container_width=True):
        health = api_request("GET", "/health")
        if health is not None: st.success(f"Backend status: {health.get('status', 'Connected')}")

if LOGO_PATH.exists(): st.image(str(LOGO_PATH), width=240)
st.markdown("""<div class="hero"><h1>Lumi-It</h1><p><strong>GUIDE • LEARN • GROW</strong></p>
<p>Your personal reading plan. Practise one activity each day, read carefully, and track your progress.</p></div>""", unsafe_allow_html=True)
st.caption("Research prototype for adaptive reading assistance. This system supports learning and does not diagnose dyslexia.")

profile_tab, progress_tab = st.tabs(["🗓️ My Weekly Plan", "📈 My Progress"])

with profile_tab:
    st.subheader("Learner Profile")
    st.write("Enter the latest assessment information and preferences to generate one personalized 7-day plan.")
    with st.form("learner_weekly_plan_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            learner_id = st.text_input("Learner ID", value=st.session_state.get("learner_id", "DEMO_LEARNER_01")).strip()
            age = st.number_input("Age", min_value=8, max_value=10, value=8, step=1)
            grade = st.number_input("Grade", min_value=3, max_value=5, value=3, step=1)
            reading_score = st.slider("Reading score (%)", 0, 100, 65)
            reading_speed = st.number_input("Reading speed (words per minute)", 0, 500, 60)
        with c2:
            pronunciation = st.slider("Pronunciation accuracy (%)", 0, 100, 70)
            fluency = st.slider("Fluency score (%)", 0, 100, 65)
            difficulty = st.selectbox("Current reading difficulty", ["Easy", "Medium", "Hard"], index=1)
            # Display labels clarify that these are target skills, while the values
            # sent to the backend retain the existing catalogue/model terminology.
            weak_skill_labels = {
                "Phonics (target skill)": "Phonics",
                "Word Recognition (target skill)": "Word Recognition",
                "Fluency (target skill)": "Fluency",
                "Comprehension (target skill)": "Comprehension",
                "Pronunciation Accuracy (target skill)": "Pronunciation",
                "Vocabulary (target skill)": "Vocabulary",
            }
            weak_skill_label = st.selectbox("Main skill to practise", list(weak_skill_labels.keys()))
            weak_skill = weak_skill_labels[weak_skill_label]
            st.info(
                "Previous activity score is taken from saved learner progress. "
                "For a new learner with no saved results, no previous score is used."
            )
        with c3:
            topic = st.selectbox("Favourite topic", ["Animals", "Stories", "Space"])
            activity_type = st.selectbox(
                "Preferred activity type",
                ["Word Games", "Reading Stories", "Pronunciation", "Comprehension"],
            )
            duration = st.select_slider("Preferred activity duration (minutes)", options=[10, 15, 20], value=15)
            st.caption("Completion rate is calculated from saved progress. A new learner starts at 0%.")
        week_number = st.selectbox("Plan week", [1, 2, 3, 4], index=0)
        st.caption("For Week 2 onwards, saved scores and completed activities should guide adaptation. Keep the same learner ID to use their progress.")
        generate = st.form_submit_button("Generate My 7-Day Plan", type="primary", use_container_width=True)

    if generate:
        if not learner_id:
            st.warning("Please enter a learner ID.")
        else:
            features = {
                "learner_id": learner_id, "age": int(age), "grade": int(grade),
                "preferred_topic": topic, "preferred_activity": activity_type,
                "preferred_duration_min": int(duration), "reading_score": float(reading_score),
                "reading_speed_wpm": float(reading_speed), "pronunciation_accuracy": float(pronunciation),
                "fluency_score": float(fluency), "difficulty_level": difficulty,
                "difficulty_score": {"Easy": 0.25, "Medium": 0.5, "Hard": 0.75}[difficulty],
                "weak_skill": weak_skill, "previous_score": None,
                "activity_completion_rate": 0.0,
                # Kept as a neutral value for compatibility with the current backend schema.
                "previous_activity": "None",
            }
            with st.spinner("Creating your personalized plan..."):
                data = api_request("POST", "/weekly-plans", json={"learner_id": learner_id, "week_number": int(week_number), "learner_features": features})
            if data is not None:
                st.session_state["weekly_plan"] = data
                st.session_state["weekly_plan_learner_id"] = learner_id
                st.session_state["weekly_plan_week"] = int(week_number)
                st.session_state["learner_id"] = learner_id
                st.success("Your weekly plan is ready!")

    if "weekly_plan" in st.session_state:
        data = st.session_state["weekly_plan"]
        learner = st.session_state.get("weekly_plan_learner_id", "")
        week = st.session_state.get("weekly_plan_week", 1)
        activities = plan_activities(data)
        st.divider()
        st.subheader(f"Your Week {week} Reading Plan")
        if not activities:
            st.warning("The backend response has no activities list. Ensure POST /weekly-plans returns an 'activities' array.")
        else:
            st.write("Complete one activity each day. Your results can help personalize future plans.")
            for pos, activity in enumerate(activities[:7], 1):
                day = activity.get("day_number", pos)
                aid = activity_id_of(activity)
                title = activity.get("title", "Reading activity")
                with st.expander(f"Day {day}: {title} · {activity.get('duration_minutes', 10)} minutes", expanded=(pos == 1)):
                    st.write(f"**Activity ID:** {aid}")
                    st.write(f"**Type:** {activity.get('activity_type', 'Activity')}")
                    st.write(f"**Target skill:** {activity.get('target_skill', 'Reading skills')}")
                    st.write(f"**Difficulty:** {activity.get('difficulty_level', 'Not specified')}")
                    st.write(f"**Duration:** {activity.get('duration_minutes', 10)} minutes")
                    st.write(f"**Learning objective:** {activity.get('learning_objective', 'Practise the target reading skill.')}")
                    if aid and aid not in ACTIVITY_CONTENT and not activity.get("passage") and not activity.get("questions"):
                        st.caption("Detailed story/quiz content has not been configured for this activity yet.")
                    show_activity(learner, activity)

with progress_tab:
    st.subheader("Learning Progress")
    progress_id = st.text_input(
        "Learner ID",
        value=st.session_state.get("weekly_plan_learner_id", st.session_state.get("learner_id", "DEMO_LEARNER_01")),
        key="progress_learner_id",
    ).strip()
    if st.button("Load My Progress", type="primary", use_container_width=True):
        if not progress_id:
            st.warning("Please enter a learner ID.")
        else:
            data = api_request("GET", f"/progress/{progress_id}")
            if data is not None:
                st.session_state["learner_progress"] = data
                st.session_state["learner_progress_id"] = progress_id
                # Refresh plan metadata whenever progress is reloaded.
                plan_lookup = {}
                for week_num in range(1, 5):
                    try:
                        response = requests.get(
                            f"{API_BASE_URL}/weekly-plans/{progress_id}/{week_num}", timeout=10
                        )
                        if response.ok:
                            plan_data = response.json()
                            for position, activity in enumerate(plan_activities(plan_data), 1):
                                activity_id = activity_id_of(activity)
                                if activity_id:
                                    plan_lookup.setdefault(activity_id, {
                                        "week": week_num,
                                        "day": activity.get("day_number", position),
                                        "title": activity.get("title", "Reading activity"),
                                        "activity_type": activity.get("activity_type", ""),
                                    })
                    except (requests.RequestException, ValueError):
                        # Missing weeks are normal; progress still displays and exports.
                        continue
                st.session_state["learner_plan_lookup"] = plan_lookup

    if "learner_progress" in st.session_state:
        data = st.session_state["learner_progress"]
        shown_id = st.session_state.get("learner_progress_id", progress_id)
        # Do not accidentally show one learner's cached progress after changing the ID.
        if shown_id != progress_id:
            st.info("Click **Load My Progress** to load progress for this learner ID.")
        else:
            stats = data.get("stats", data)
            a, b, c = st.columns(3)
            a.metric("Activities completed", stats.get("completed_activities", stats.get("completed_count", 0)))
            b.metric("Average score", f"{float(stats.get('average_score', 0) or 0):.1f}%")
            c.metric("Total attempts", stats.get("total_attempts", stats.get("attempts", 0)))
            records = data.get("records", data.get("activities", []))
            plan_lookup = st.session_state.get("learner_plan_lookup", {})
            export_rows = []
            if isinstance(records, list) and records:
                st.markdown("#### Activity history by week")
                for item in records:
                    activity_id = str(item.get("item_id", item.get("activity_id", "Activity")))
                    plan_info = plan_lookup.get(activity_id, {})
                    week_label = f"Week {plan_info['week']}" if plan_info.get("week") else "Week not identified"
                    day_label = f"Day {plan_info['day']}" if plan_info.get("day") else ""
                    title = plan_info.get("title", item.get("title", "Activity"))
                    score = item.get("score", "N/A")
                    completed = bool(item.get("completed"))
                    with st.container(border=True):
                        st.write(f"**{week_label} · {day_label} — {title}**")
                        st.caption(f"Activity ID: {activity_id} · Score: {score}% · Completed: {'Yes' if completed else 'No'}")
                    export_rows.append({
                        "learner_id": shown_id,
                        "week": plan_info.get("week", "Not identified"),
                        "day": plan_info.get("day", ""),
                        "activity_id": activity_id,
                        "activity_title": title,
                        "activity_type": plan_info.get("activity_type", item.get("activity_type", "")),
                        "score_percent": score,
                        "completed": "Yes" if completed else "No",
                    })
                csv_buffer = io.StringIO(newline="")
                fieldnames = ["learner_id", "week", "day", "activity_id", "activity_title", "activity_type", "score_percent", "completed"]
                writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_rows)
                st.download_button(
                    "Download Progress as CSV",
                    data="\ufeff" + csv_buffer.getvalue(),
                    file_name=f"{shown_id}_learning_progress.csv",
                    mime="text/csv; charset=utf-8",
                    use_container_width=True,
                )
                st.caption("Week labels are matched against the learner's saved weekly plans. Records not found in a saved plan are marked 'Week not identified'.")
            else:
                st.info("No saved activity results found for this learner yet.")

st.divider()
st.caption("The weekly plan requires POST /weekly-plans. Progress uses POST /progress/record and GET /progress/{learner_id}.")

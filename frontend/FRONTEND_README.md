# Lumi-It dashboard (Component 02)

Run (Docker):

    docker run -it --rm -p 8501:8501 -v "${PWD}:/workspace" -w /workspace dyslexia-ai \
      streamlit run frontend/app.py --server.address 0.0.0.0 --server.headless true

Then open http://localhost:8501. Before the first run, fill the log from the synthetic Component 01 payloads:

    python run_demo_payloads.py

## What changed (Changes 2, 3 and 4)
- Child / parent view: ONE page with the whole cognitive profile (learner + session, difficulty level / score /
  certainty, strengths, weak skills, cognitive profile with explanations, support needed). No "Next" pages.
- Support intensity is only one label: Low / Medium / High support required. Recommendations are not shown
  (Component 03 owns them).
- Demo role selector on the first screen: log in as Child A / B / C or as the teacher. Not real authentication.
- Teacher view: Students -> Sessions of the selected student -> cognitive profile of that session ->
  Component 04 placeholder box (no invented results).

## Tests

    python test_dashboard.py     # Part A pure Python, Part B drives the real app with Streamlit AppTest

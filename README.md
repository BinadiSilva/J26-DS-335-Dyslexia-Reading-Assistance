# Dyslexia Reading Assistance

This project trains the reading-difficulty models from the raw reading-session CSV, generates synthetic demo payloads, replays them through the Component 1 -> Component 2 pipeline, and launches the Streamlit UI.

> Note: the `outputs/` directory is gitignored, so on a fresh clone there is no trained model yet. Run the steps below in order to build the artifacts locally.

## 1) Train the model

From the repo root, build the Docker image if needed and train the pipeline:

```bash
docker build -t dyslexia-ai .
docker --context desktop-linux run --rm -v "${PWD}:/workspace" -w /workspace dyslexia-ai python -c "from backend.app.services.reading_difficulty.pipeline import train_pipeline; train_pipeline('datasets/comp02_raw_reading_sessions.csv')"
```

This creates `outputs/artifacts.joblib`.

## 2) Generate demo payloads

```bash
docker --context desktop-linux run --rm -v "${PWD}:/workspace" -w /workspace dyslexia-ai python make_demo_payloads.py
```

This writes the synthetic Component 1 payloads into `demo_payloads/`.

## 3) Run the demo payloads through validation and prediction

```bash
docker --context desktop-linux run --rm -v "${PWD}:/workspace" -w /workspace dyslexia-ai python run_demo_payloads.py
```

This writes the demo session log to `outputs/session_log.jsonl`.

## 4) Launch the UI

```bash
docker --context desktop-linux run --rm -p 8501:8501 -v "${PWD}:/workspace" -w /workspace dyslexia-ai streamlit run frontend/frontend/app.py --server.address 0.0.0.0 --server.headless true
```

Then open the app in a browser at:

```text
http://localhost:8501
```

## Typical local workflow

```bash
docker build -t dyslexia-ai .
docker --context desktop-linux run --rm -v "${PWD}:/workspace" -w /workspace dyslexia-ai python -c "from backend.app.services.reading_difficulty.pipeline import train_pipeline; train_pipeline('datasets/comp02_raw_reading_sessions.csv')"
docker --context desktop-linux run --rm -v "${PWD}:/workspace" -w /workspace dyslexia-ai python make_demo_payloads.py
docker --context desktop-linux run --rm -v "${PWD}:/workspace" -w /workspace dyslexia-ai python run_demo_payloads.py
docker --context desktop-linux run --rm -p 8501:8501 -v "${PWD}:/workspace" -w /workspace dyslexia-ai streamlit run frontend/frontend/app.py --server.address 0.0.0.0 --server.headless true
```

# Lumi-It frontend update (replace the old frontend)

1. Adds
   .streamlit/config.toml (forces the light theme) and test_dashboard.py.
   Keep your existing frontend/demo_children.json and outputs/ folder.
2. In PowerShell (repo root):
   docker run -it --rm -p 8501:8501 -v "${PWD}:/workspace" -w /workspace dyslexia-ai streamlit run frontend/app.py --server.address 0.0.0.0 --server.headless true
3. Open http://localhost:8501
4. Check: docker run ... bash, then: python test_dashboard.py

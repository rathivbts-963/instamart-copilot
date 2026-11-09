#!/usr/bin/env bash
# Runs the whole Copilot pipeline in order (activate the virtual environment first).
set -e
python generate_data.py
python src/a1_audit.py
python src/a2_clean.py
python src/a3_validate.py
python src/a4_sql.py
python src/a5_eda.py
python src/b1_segments.py
python src/b2_lapse.py
python src/c_triage.py
python src/d_rag.py
python src/d_agent.py

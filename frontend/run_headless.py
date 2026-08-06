from pathlib import Path
import streamlit.web.bootstrap as bs

script = str(Path(__file__).resolve().parent / "streamlit_app.py")
# Run programmatically to avoid CLI first-run prompt
bs.run(script, False, [script, "--server.port", "8502"], {})

"""Launch the FastAPI server with the right import path.

    python scripts/run_api.py [--port 8000]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description="Run the CineMatch API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("cinematch.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

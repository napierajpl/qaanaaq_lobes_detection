#!/usr/bin/env python3
"""Script to start MLflow UI server."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="Start MLflow UI server")
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to run MLflow UI on (default: 5001)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    # Get mlruns directory
    mlruns_dir = project_root / "mlruns"

    if not mlruns_dir.exists():
        print(f"Error: mlruns directory not found at {mlruns_dir}")
        print("Run a training script first to create MLflow runs.")
        sys.exit(1)

    # Convert path to file:// URI format for MLflow (handles Windows paths correctly)
    mlruns_uri = mlruns_dir.as_uri()

    print(f"Starting MLflow UI on http://{args.host}:{args.port}")
    print(f"Tracking URI: {mlruns_uri}")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)

    # Start MLflow UI
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--port",
                str(args.port),
                "--host",
                args.host,
                "--backend-store-uri",
                mlruns_uri,
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\n\nMLflow UI server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\nError starting MLflow UI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

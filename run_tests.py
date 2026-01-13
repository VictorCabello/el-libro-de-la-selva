"""Run pytest with coverage."""

import sys
import subprocess

if __name__ == "__main__":
    sys.exit(subprocess.run(["./venv/bin/pytest", "--cov=el_libro_de_la_selva", "tests/"]).returncode)

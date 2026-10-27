"""Run meaningful checks and record their exact source fingerprint."""
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.evaluate import code_fingerprint


def main():
    tests = subprocess.run([sys.executable, "-m", "pytest", "backend/tests", "-q"], cwd=ROOT, text=True, capture_output=True)
    print(tests.stdout, end="")
    print(tests.stderr, end="", file=sys.stderr)
    if tests.returncode:
        raise SystemExit(tests.returncode)
    build = subprocess.run(["npm", "run", "build"], cwd=ROOT / "frontend", text=True, capture_output=True)
    print(build.stdout, end="")
    print(build.stderr, end="", file=sys.stderr)
    if build.returncode:
        raise SystemExit(build.returncode)
    match = re.search(r"(\d+) passed", tests.stdout)
    result = {"code_sha256": code_fingerprint(ROOT), "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "tests_passed": int(match.group(1)), "pytest": "passed", "frontend_build": "passed"}
    (ROOT / "benchmarks/checks.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

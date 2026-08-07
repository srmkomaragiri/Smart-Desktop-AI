"""Wrapper to run verify_all.py and capture output to a file."""
import subprocess
import sys
import os

result = subprocess.run(
    [sys.executable, "training/verify_all.py"],
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)
    f.write(f"\n=== EXIT CODE: {result.returncode} ===\n")

print(f"Results written to {output_path}")
sys.exit(result.returncode)

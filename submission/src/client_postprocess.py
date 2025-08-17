"""
Dummy output for the workload, not accounting for test size.
"""

from pathlib import Path

def main():

    OUT_PATH = Path("io/result.txt")
    OUT_PATH.write_text(f"0\n", encoding="utf-8")

if __name__ == "__main__":
    main()
"""Starter optimization experiment template.

Copy this file into a run folder before modifying it. Generated experiments
should write results.csv, figures, and a final SUMMARY_JSON line to stdout.
"""

from pathlib import Path


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    print({"status": "template_only", "output_dir": str(out_dir)})


if __name__ == "__main__":
    main()


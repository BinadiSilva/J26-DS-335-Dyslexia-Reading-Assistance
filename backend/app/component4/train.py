"""CLI: compare all proposal models on synthetic or supplied session records."""
import argparse
import json
import pandas as pd

from .models import train_comparison
from .synthetic import generate_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--csv", help="Validated session contract; never a BAS-to-fluency conversion")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 1000:
        parser.error("epochs must be between 1 and 1000")
    frame = generate_cohort() if args.demo else pd.read_csv(args.csv)
    _, report = train_comparison(frame, args.output, epochs=args.epochs)
    print(json.dumps({"source": report["source"], "selected_model": report["selected_model"],
                      "test": report["selected_refit_test"], "stagnation": report["stagnation"]}, indent=2))


if __name__ == "__main__":
    main()

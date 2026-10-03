# esrs/run.py
"""Entry point: `python -m esrs.run [--use-amazon]`."""
import argparse
from .experiment import main


def cli():
    ap = argparse.ArgumentParser(description="ESRS offline experiment")
    ap.add_argument("--use-amazon", action="store_true",
                    help="Stream real Amazon Reviews 2023 metadata "
                         "(falls back to synthetic on failure).")
    args = ap.parse_args()
    main(use_amazon=args.use_amazon)


if __name__ == "__main__":     # still works with `python -m esrs.run`
    cli()

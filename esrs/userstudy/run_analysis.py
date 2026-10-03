"""CLI: analyze a collected user-study DB.

    python -m esrs.userstudy.run_analysis --db esrs_userstudy.sqlite
    python -m esrs.userstudy.run_analysis --db esrs_userstudy_sim.sqlite
"""
import argparse
from .analysis import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="esrs_userstudy.sqlite")
    args = ap.parse_args()
    main(args.db)
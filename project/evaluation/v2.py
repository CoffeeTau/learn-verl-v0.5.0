"""Export V2 and evaluate clean dev against V0/V1."""
from project.evaluation.v1 import main

if __name__ == "__main__":
    raise SystemExit(main(version="v2"))

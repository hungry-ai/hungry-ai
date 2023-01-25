import argparse
from pathlib import Path


def main(numwords: int, output: Path) -> None:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--numwords", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    main(args.numwords, args.output)

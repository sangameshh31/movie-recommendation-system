"""Download the MovieLens dataset."""

import argparse

from cinematch.config import SETTINGS
from cinematch.data import download


def main():
    parser = argparse.ArgumentParser(description="Download MovieLens data.")
    parser.add_argument("--size", default=SETTINGS.data.size, choices=["100k", "1m", "25m"])
    args = parser.parse_args()
    dest = download(size=args.size)
    print(f"Dataset ready at: {dest}")


if __name__ == "__main__":
    main()

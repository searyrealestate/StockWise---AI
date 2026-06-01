"""CLI entry point. Phase 1 implements only --version."""

import argparse
import sys

from micha7 import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="micha7", description="micha7_analyzer CLI")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        print(f"micha7_analyzer {__version__}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

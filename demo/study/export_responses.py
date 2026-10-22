"""Export the append-only local study database to CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .core import DATABASE_PATH, export_responses
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from study.core import DATABASE_PATH, export_responses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(export_responses(args.database, args.output))


if __name__ == "__main__":
    main()

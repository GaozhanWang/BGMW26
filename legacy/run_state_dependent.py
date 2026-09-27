# Copyright (c) 2026 Gaozhan Wang. See LICENSE.
# Modification note: lightweight CLI preserving the historical state_dependent settings.
"""Run the historical state dependent uncorrected stopped-BEL experiment."""


def main(argv=None):
    if __package__:
        from ._entrypoint import run
    else:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from legacy._entrypoint import run
    return run("state_dependent", argv)


if __name__ == "__main__":
    raise SystemExit(main())

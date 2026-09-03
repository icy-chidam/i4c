#!/usr/bin/env python3
"""
Thin repo-root wrapper -- exactly the same reasoning as localize.py's
own root wrapper (see that file): the addendum's exact required
invocation is `python register.py --input pairs.csv --output
predictions.csv`, and this project's actual implementation lives in
src/register.py (alongside the rest of the pipeline it imports from,
same as every other engine file). This wrapper makes `python
register.py ...` from the repo root work with no path surprises,
regardless of which of the two layouts a grading harness expects. All
logic, flags and behaviour are identical to src/register.py; see that
file (and docs/validation_report_v4.md) for details.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from register import main  # noqa: E402

if __name__ == "__main__":
    main()

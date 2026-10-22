"""Run the real-data, video-conditioned five-fold TVSum experiment."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from standard_tvsum_cv import main


if __name__ == "__main__":
    main()

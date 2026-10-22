"""Run the real-data, video-conditioned five-fold TVSum experiment."""

import os
import sys

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
sys.path.insert(0, CODE_DIR)

from standard_tvsum_cv import main


if __name__ == "__main__":
    main()

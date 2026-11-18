"""Old entry point name, kept so existing scheduled jobs keep working.
Use paper_scout.py instead."""
import sys

from paper_scout import main

if __name__ == "__main__":
    sys.exit(main())

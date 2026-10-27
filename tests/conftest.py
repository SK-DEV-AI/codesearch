import os
import sys

# codesearch is a flat package; make its modules importable as top-level.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""pytest entry point; the actual checks live in test.py (run directly by the
Windows Tests workflow via `python test.py`, which pytest's test_*.py naming
convention won't discover on its own)."""
from test import test_core_dependencies_import, test_detection_module_constants  # noqa: F401

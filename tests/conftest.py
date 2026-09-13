import sys
from pathlib import Path

import pytest

# Make tests/helpers importable (fake backend entrypoints resolve through it).
TESTS_ROOT = Path(__file__).parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))


@pytest.fixture(autouse=True)
def _no_background_round_evaluation(monkeypatch):
    """A finished loop round evaluates its model on a daemon thread; tests
    would race it (records rewritten mid-assertion, work after teardown).
    Tests that want the evaluation call evaluate_round_splits themselves or
    re-patch this to run inline."""
    import horos.api.loop as loop_mod

    monkeypatch.setattr(loop_mod, "_evaluate_in_background", lambda project, number: None)

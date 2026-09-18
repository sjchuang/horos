"""UTF-8 text streams on every platform (R7).

Windows picks a legacy code page for `sys.stdout` whenever it is a pipe or a
file — cp950 on a Traditional Chinese install — so printing anything outside
that page raises UnicodeEncodeError and kills the process. We do not control
what a backend prints: rfdetr draws its per-epoch metrics table with box
characters, so a training worker whose stdout is redirected into worker.log
died with `'cp950' codec can't encode character U+250F` (a box corner) as soon as the
first validation table was drawn, before any checkpoint existed.

Two halves, both needed:

- `use_utf8_streams()` fixes the streams of the process that calls it, which
  is what an entry point (`horos ...`, `python -m horos.api.train_worker`)
  can do for itself.
- `child_env()` fixes every process we spawn. It has to be the environment
  and not a reconfigure, because the interpreters started under `spawn` —
  DataLoader workers, a backend's own helpers — are fresh processes that read
  PYTHONIOENCODING at startup and never see the parent's reconfigure.

`backslashreplace` rather than `strict`: a lone surrogate from a path decoded
with `surrogateescape` is still unencodable as UTF-8, and losing one filename
to an escape sequence in a log beats losing the run.
"""

from __future__ import annotations

import os
import sys

#: what every horos process should use for its own stdout/stderr
STREAM_ENCODING = "utf-8"
STREAM_ERRORS = "backslashreplace"

_CHILD_DEFAULTS = {
    "PYTHONIOENCODING": f"{STREAM_ENCODING}:{STREAM_ERRORS}",
    # a log nobody can read until the process exits is no log at all, and
    # these children are watched while they run (worker.log, serve.log)
    "PYTHONUNBUFFERED": "1",
}


def use_utf8_streams() -> None:
    """Make this process's stdout/stderr encode UTF-8, whatever the locale."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # replaced by a capture buffer (pytest) or not a TextIO
        try:
            reconfigure(encoding=STREAM_ENCODING, errors=STREAM_ERRORS)
        except (ValueError, OSError):  # detached, closed, already written to
            pass


def child_env(**extra: str) -> dict[str, str]:
    """This process's environment plus the stream settings a child needs.

    The defaults win over an inherited PYTHONIOENCODING on purpose — an
    ambient cp950 is exactly the setting that breaks the child — while
    `extra` wins over both, for a caller that needs something specific.
    """
    return {**os.environ, **_CHILD_DEFAULTS, **extra}

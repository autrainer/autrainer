from contextlib import contextmanager
import io
import sys
from typing import Generator


@contextmanager
def silence() -> Generator[None, None, None]:
    """Context manager to suppress stdout and stderr.

    Yields:
        None.
    """
    new_stdout, new_stderr = io.StringIO(), io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_stdout, new_stderr
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

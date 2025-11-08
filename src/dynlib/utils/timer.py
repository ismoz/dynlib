from time import perf_counter
from contextlib import ContextDecorator

class Timer(ContextDecorator):
    def __init__(self, label: str | None = None, printer=print):
        self.label = f"{label}: " if label else ""
        self.printer = printer
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = perf_counter() - self._t0
        self.printer(f"{self.label}{dt*1000:.3f} ms")
        return False  # don't suppress exceptions

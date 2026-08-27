"""A progress line for the steps that are slow enough to wonder about.

Two things in this pipeline take long enough that the user needs to know
whether to wait or go for coffee: downloading a few thousand DEM tiles, and
reading heights back out of them. Both are loops over a known number of items,
which is all it takes to turn "something is happening" into "eleven minutes
left".

The report goes to stderr, where the log lines already go, and adapts to what
is reading it. On a terminal it rewrites one line in place, so a long loop
costs a single line of scrollback. Redirected to a file or a CI log, where a
carriage return is just a character, each update is an ordinary log record
instead.

Updates are throttled by time rather than by count, because the right count
depends on how slow each item is -- every 25 tiles is far too chatty for a
cached read and far too quiet for a download over a slow link. The first and
last update always come out.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import IO, Iterable, Iterator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Seconds between updates. Long enough not to scroll a log, short enough that
#: a stalled run is visibly stalled.
UPDATE_INTERVAL_S = 2.0


def format_duration(seconds: float) -> str:
    """A duration a human can read at a glance, to two units at most."""
    if seconds < 0 or not (seconds == seconds):  # negative or NaN
        return "?"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


class Progress:
    """
    Count towards a known total, reporting at most every ``interval_s``.

    ``advance`` is called once per item done; ``finish`` closes the line off
    with the total and how long it took. Both are cheap enough to call in a
    tight loop -- all that happens most times is a clock read and a comparison.

    Use it as a context manager to be sure the line is closed even if the loop
    raises, or call :func:`track` to wrap an iterable in one.
    """

    def __init__(
        self,
        total: int,
        label: str,
        *,
        interval_s: float = UPDATE_INTERVAL_S,
        stream: IO[str] | None = None,
    ) -> None:
        self.total = max(0, int(total))
        self.label = label
        self.interval_s = interval_s
        self._stream: IO[str] = sys.stderr if stream is None else stream
        self._tty = bool(getattr(self._stream, "isatty", lambda: False)())
        self.done = 0
        self._started = time.monotonic()
        self._last_report = self._started
        self._width = 0
        self._finished = False

    # -- the loop calls these ---------------------------------------------

    def advance(self, n: int = 1) -> None:
        self.done += n
        now = time.monotonic()
        if now - self._last_report < self.interval_s:
            return
        self._last_report = now
        self._emit(self._line(now))

    def finish(self) -> None:
        """Report the total and the elapsed time, once."""
        if self._finished:
            return
        self._finished = True
        elapsed = time.monotonic() - self._started
        self._emit(
            f"{self.label}: {self.done:,} of {self.total:,} in "
            f"{format_duration(elapsed)}",
            last=True,
        )

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exception: object) -> None:
        self.finish()

    # -- rendering --------------------------------------------------------

    def _line(self, now: float) -> str:
        elapsed = now - self._started
        share = self.done / self.total if self.total else 1.0
        line = f"{self.label}: {self.done:,}/{self.total:,} ({share:.0%})"
        if self.done and self.done < self.total:
            remaining = elapsed / self.done * (self.total - self.done)
            line += f", {format_duration(remaining)} left"
        return line

    def _emit(self, line: str, last: bool = False) -> None:
        if not self._tty:
            logger.info("%s", line)
            return
        # Pad to whatever the previous line needed, so a shortening line (100%
        # after 99%, or the final one) does not leave its own tail behind.
        padding = " " * max(0, self._width - len(line))
        self._width = len(line)
        self._stream.write(f"\r{line}{padding}" + ("\n" if last else ""))
        self._stream.flush()


def track(
    items: Iterable[T],
    total: int,
    label: str,
    *,
    interval_s: float = UPDATE_INTERVAL_S,
) -> Iterator[T]:
    """Yield from ``items``, counting each one towards ``total``."""
    with Progress(total, label, interval_s=interval_s) as progress:
        for item in items:
            yield item
            progress.advance()


__all__ = ["Progress", "UPDATE_INTERVAL_S", "format_duration", "track"]

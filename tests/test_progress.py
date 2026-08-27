"""The progress line: what it says, how often, and where it says it."""

from __future__ import annotations

import io
import logging

import pytest

from valma_bike_and_walk.progress import Progress, format_duration, track


class FakeTerminal(io.StringIO):
    """A stream that claims to be a terminal, so the bar path is taken."""

    def isatty(self) -> bool:
        return True


@pytest.mark.parametrize(
    "seconds, expected",
    [(0, "0s"), (45, "45s"), (61, "1m 01s"), (3_600, "1h 00m"), (7_000, "1h 56m")],
)
def test_durations_read_at_a_glance(seconds, expected):
    assert format_duration(seconds) == expected


def test_a_terminal_gets_one_line_that_is_rewritten():
    stream = FakeTerminal()
    with Progress(4, "tiles", interval_s=0.0, stream=stream) as progress:
        for _ in range(4):
            progress.advance()

    written = stream.getvalue()
    assert written.count("\n") == 1  # only the closing line ends one
    assert written.startswith("\r")
    assert written.endswith("tiles: 4 of 4 in 0s\n")


def test_a_shortening_line_does_not_leave_its_tail_behind():
    stream = FakeTerminal()
    progress = Progress(1_000, "tiles", interval_s=0.0, stream=stream)
    progress.advance()
    long_line = stream.getvalue()
    progress.finish()

    last = stream.getvalue()[len(long_line) :]
    assert len(last.rstrip("\n")) >= len(long_line)  # padded over the old text


def test_a_log_file_gets_records_rather_than_carriage_returns(caplog):
    stream = io.StringIO()  # not a terminal
    with caplog.at_level(logging.INFO, logger="valma_bike_and_walk.progress"):
        with Progress(2, "tiles", interval_s=0.0, stream=stream) as progress:
            progress.advance()
            progress.advance()

    assert stream.getvalue() == ""
    assert "\r" not in caplog.text
    assert "tiles: 2 of 2" in caplog.text


def test_updates_are_throttled_by_time(caplog):
    """A tight loop must not turn into one log line per item."""
    with caplog.at_level(logging.INFO, logger="valma_bike_and_walk.progress"):
        with Progress(500, "tiles", interval_s=3_600.0, stream=io.StringIO()) as p:
            for _ in range(500):
                p.advance()

    # Nothing but the closing line: an hour never passed.
    assert len(caplog.records) == 1
    assert "tiles: 500 of 500" in caplog.text


def test_an_estimate_appears_once_there_is_something_to_estimate_from():
    stream = FakeTerminal()
    progress = Progress(10, "tiles", interval_s=0.0, stream=stream)
    progress.advance()
    assert "left" in stream.getvalue()


def test_track_counts_what_it_yields(caplog):
    with caplog.at_level(logging.INFO, logger="valma_bike_and_walk.progress"):
        assert list(track(iter("abc"), 3, "letters", interval_s=0.0)) == list("abc")
    assert "letters: 3 of 3" in caplog.text


def test_the_line_is_closed_even_when_the_loop_raises():
    stream = FakeTerminal()
    with pytest.raises(ZeroDivisionError):
        with Progress(3, "tiles", interval_s=0.0, stream=stream) as progress:
            progress.advance()
            raise ZeroDivisionError

    assert stream.getvalue().rstrip().endswith("tiles: 1 of 3 in 0s")
    assert stream.getvalue().endswith("\n")

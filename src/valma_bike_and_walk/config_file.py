"""Command settings from TOML files, layered under the command line.

Any option a command takes can be written in a file instead, named as on the
command line without the leading dashes -- ``max_minutes`` or ``max-minutes``
for ``--max-minutes``::

    # scenarios/base.toml
    mode = "bike"
    links = "../output/{mode}_links.gpkg"
    centroids = "../data/centroids.csv"
    id_column = "zone_id"

    [assign]
    demand = "../data/demand.omx"
    demand_matrix = "bike"

    # scenarios/2040.toml
    extends = "base.toml"
    scenario = "2040"
    demand = "../data/demand_2040.omx"

    valma assign -c scenarios/2040.toml --gpkg

How the layers stack, lowest first:

1. the command's own defaults;
2. the files: each file's ``extends`` before the file itself, and several
   ``-c`` in the order given. A later file wins over an earlier one, and
   within a file a ``[command]`` section wins over the top level;
3. the command line.

A top-level key goes to every command that has that option and is ignored by
the rest, so one file serves ``matrix``, ``assign`` and ``view`` alike. A key
no command has, a section that is not a command, or a value of the wrong type
is an error: a typo would otherwise be a setting silently not applied.

Relative paths are taken from the file they are written in, not from where
valma happens to be run. ``{mode}`` and ``{scenario}`` in a value are filled
in from the final settings.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib

#: Key naming files to read first, as a base the naming file overrides.
EXTENDS = "extends"

#: Not settings: argparse's own help, and the options that pick the files.
_NOT_SETTINGS = frozenset({"help", "config", "show_config"})

#: About this invocation rather than its result, so left out of a record.
#: ``scenario`` is already folded into ``output_dir`` by then.
_NOT_RECORDED = frozenset({"force_reload", "scenario", "verbose"})

#: Shared by several commands but meaning something different in each.
_NEVER_SHARED = frozenset({"out"})


class ConfigError(ValueError):
    """A settings file that cannot be applied as written."""


@dataclass(frozen=True)
class Setting:
    """One value from one file."""

    value: Any
    source: Path
    #: (layer, from a section): the higher one wins.
    rank: tuple[int, bool]


@dataclass
class Config:
    """Every file's settings, merged; later layers have replaced earlier ones."""

    files: list[Path] = field(default_factory=list)
    shared: dict[str, Setting] = field(default_factory=dict)
    sections: dict[str, dict[str, Setting]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def load(paths: Iterable[Path | str]) -> Config:
    """Read settings files in order, each on top of the ones before it."""
    config = Config()
    for path in paths:
        _load_into(config, Path(path), ())
    return config


def _load_into(config: Config, path: Path, chain: tuple[Path, ...]) -> None:
    path = Path(os.path.abspath(path))
    if path in chain:
        loop = " -> ".join(str(p) for p in (*chain, path))
        raise ConfigError(f"Settings files extend each other in a loop: {loop}")
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        where = f" (extended by {chain[-1]})" if chain else ""
        raise ConfigError(f"No such settings file: {path}{where}") from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: {error}") from None

    data = _normalised(data, path, None)
    bases = data.pop(EXTENDS, [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(b, str) for b in bases):
        raise ConfigError(f"{path}: {EXTENDS} must be a file name or a list of them.")
    for base in bases:
        _load_into(config, path.parent / base, (*chain, path))

    layer = len(config.files)
    config.files.append(path)
    for key, value in data.items():
        if not isinstance(value, dict):
            config.shared[key] = Setting(value, path, (layer, False))
            continue
        section = config.sections.setdefault(key, {})
        for option, inner in _normalised(value, path, key).items():
            if isinstance(inner, dict):
                raise ConfigError(
                    f"{path}: [{key}.{option}] -- sections are commands, and "
                    "they do not nest."
                )
            section[option] = Setting(inner, path, (layer, True))


def _normalised(table: dict, path: Path, section: str | None) -> dict:
    """Keys as argparse names them: ``max-minutes`` and ``max_minutes`` alike."""
    out: dict = {}
    for key, value in table.items():
        name = key.replace("-", "_")
        if name in out:
            where = f"[{section}] " if section else ""
            raise ConfigError(f"{path}: {where}{name!r} is given twice.")
        out[name] = value
    return out


# --------------------------------------------------------------------------
# Applying to the parser
# --------------------------------------------------------------------------


def subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """The parser of each command, by name."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def options(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """A parser's settable options, by destination name."""
    return {
        action.dest: action
        for action in parser._actions
        if action.dest not in _NOT_SETTINGS
        and action.dest != argparse.SUPPRESS
        and not isinstance(action, argparse._SubParsersAction)
    }


def check(config: Config, parser: argparse.ArgumentParser) -> None:
    """Refuse anything in the files that no command would take as written."""
    commands = subparsers(parser)
    by_command = {name: options(sub) for name, sub in commands.items()}
    top = options(parser)
    known = set(top).union(*by_command.values())

    for key, setting in config.shared.items():
        if key not in known:
            raise ConfigError(_unknown(setting.source, key, known, "No command takes"))
        for opts in (top, *by_command.values()):
            if key in opts:
                _convert(opts[key], key, setting)

    for name, section in config.sections.items():
        if name not in commands:
            first = next(iter(section.values()), None)
            source = first.source if first else config.files[-1]
            raise ConfigError(_unknown(source, name, commands, "There is no command"))
        opts = by_command[name]
        for key, setting in section.items():
            if key not in opts:
                raise ConfigError(
                    _unknown(setting.source, key, opts, f"valma {name} takes no")
                )
            _convert(opts[key], key, setting)


def _unknown(source: Path, key: str, choices: Iterable[str], lead: str) -> str:
    message = f"{source}: {lead} {key!r}."
    close = difflib.get_close_matches(key, list(choices), n=1)
    return message + (f" Did you mean {close[0]!r}?" if close else "")


def apply(
    config: Config, parser: argparse.ArgumentParser, command: str
) -> dict[str, Setting]:
    """
    Make the files' settings the parser's defaults for ``command``.

    The command line still overrides them, since it is parsed afterwards. An
    option the files supply is no longer required on the command line.
    Returns the settings applied, by destination name.
    """
    check(config, parser)
    sub = subparsers(parser)[command]
    applied: dict[str, Setting] = {}
    for target in (parser, sub):
        opts = options(target)
        chosen = {k: s for k, s in config.shared.items() if k in opts}
        if target is sub:
            for key, setting in config.sections.get(command, {}).items():
                if key not in chosen or setting.rank > chosen[key].rank:
                    chosen[key] = setting
        _one_per_exclusive_group(target, chosen)

        target.set_defaults(**{k: _convert(opts[k], k, s) for k, s in chosen.items()})
        for key in chosen:
            opts[key].required = False
        for group in target._mutually_exclusive_groups:
            if any(action.dest in chosen for action in group._group_actions):
                group.required = False
        applied.update(chosen)
    return applied


def _one_per_exclusive_group(
    parser: argparse.ArgumentParser, chosen: dict[str, Setting]
) -> None:
    """
    Keep only the latest of options that exclude each other.

    A base naming ``centroids`` and a scenario on top of it naming ``zones``
    mean zones: the scenario replaces the base's choice rather than clashing
    with it. Both in the same place is a mistake, as on the command line.
    """
    for group in parser._mutually_exclusive_groups:
        present = [a.dest for a in group._group_actions if a.dest in chosen]
        if len(present) < 2:
            continue
        top = max(chosen[dest].rank for dest in present)
        winners = [dest for dest in present if chosen[dest].rank == top]
        if len(winners) > 1:
            raise ConfigError(
                f"{chosen[winners[0]].source}: {' and '.join(winners)} exclude "
                "each other; give one."
            )
        for dest in present:
            if dest != winners[0]:
                del chosen[dest]


def _convert(action: argparse.Action, key: str, setting: Setting) -> Any:
    """A TOML value as the option's own type, as if typed on the command line."""
    raw = setting.value

    def fail(expected: str) -> ConfigError:
        return ConfigError(
            f"{setting.source}: {key} should be {expected}, not {raw!r}."
        )

    if action.nargs == 0:  # store_true, store_false, BooleanOptionalAction
        if not isinstance(raw, bool):
            raise fail("true or false")
        if isinstance(action, argparse._StoreFalseAction):
            return not raw
        return raw

    if action.nargs == "?" and action.const is not None:
        # Takes an optional value: true is the bare flag, false leaves it off.
        if raw is True:
            return action.const
        if raw is False:
            return None
        return _scalar(action, raw, setting, fail)

    if action.nargs in ("*", "+") or isinstance(action.nargs, int):
        if not isinstance(raw, list):
            raise fail("a list")
        if isinstance(action.nargs, int) and len(raw) != action.nargs:
            raise fail(f"a list of {action.nargs}")
        if action.nargs == "+" and not raw:
            raise fail("a list of at least one")
        return [_scalar(action, item, setting, fail) for item in raw]

    return _scalar(action, raw, setting, fail)


def _scalar(
    action: argparse.Action,
    raw: Any,
    setting: Setting,
    fail: Callable[[str], ConfigError],
) -> Any:
    kind = action.type
    if kind is Path:
        if not isinstance(raw, str):
            raise fail("a path")
        # From the file it was written in, not from wherever valma is run.
        value: Any = Path(os.path.normpath(setting.source.parent / raw))
    elif kind is float:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise fail("a number")
        value = float(raw)
    elif kind is int:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise fail("a whole number")
        value = raw
    else:
        if not isinstance(raw, str):
            raise fail("a string")
        value = raw
    if action.choices is not None and value not in action.choices:
        raise fail("one of " + ", ".join(repr(c) for c in action.choices))
    return value


# --------------------------------------------------------------------------
# After parsing
# --------------------------------------------------------------------------


def given_on_command_line(
    build_parser: Callable[[], argparse.ArgumentParser],
    argv: list[str],
    command: str,
) -> set[str]:
    """Which options ``argv`` itself sets, as opposed to defaults or files."""
    probe = build_parser()
    for target in (probe, subparsers(probe)[command]):
        for action in target._actions:
            action.default = argparse.SUPPRESS
            action.required = False
        for group in target._mutually_exclusive_groups:
            group.required = False
    return set(vars(probe.parse_args(argv)))


def settle(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    command: str,
    applied: dict[str, Setting],
    given: set[str],
    config: Config | None = None,
) -> None:
    """
    Finish what parsing alone cannot: exclusive options and placeholders.

    ``{mode}`` and ``{scenario}`` come from the command's own options where it
    has them. A command without one -- ``traffic`` takes no ``--mode`` -- still
    reads it from ``config``, so a shared ``links = "{mode}_links.gpkg"`` means
    the same layer to every command the file serves.
    """
    # --zones on the command line displaces centroids from a file.
    for group in subparsers(parser)[command]._mutually_exclusive_groups:
        dests = {action.dest for action in group._group_actions}
        if dests & given:
            for dest in dests - given:
                if dest in applied:
                    setattr(args, dest, None)

    from_files = [
        dest
        for dest in applied
        if dest not in given and getattr(args, dest, None) is not None
    ]
    mode = _not_an_option(args, config, command, "mode")
    if mode is not None:
        mode = mode.value
    scenario = _not_an_option(args, config, command, "scenario")
    if scenario is not None:
        # No option of this command holds it, so the loop below will not fill it.
        scenario = _fill(
            scenario.value,
            _Fields(
                mode=getattr(args, "mode", mode),
                scenario=None,
                dest="scenario",
                source=scenario.source,
            ),
        )

    # The scenario's own name may say which mode it is for, so fill it first.
    for dest in sorted(from_files, key=lambda d: d != "scenario"):
        fields = _Fields(
            mode=getattr(args, "mode", mode),
            scenario=getattr(args, "scenario", scenario),
            dest=dest,
            source=applied[dest].source,
        )
        setattr(args, dest, _fill(getattr(args, dest), fields))


def _not_an_option(
    args: argparse.Namespace, config: Config | None, command: str, name: str
) -> Setting | None:
    """The files' ``name``, for a command that has no option of that name."""
    if hasattr(args, name) or config is None:
        return None
    setting = config.sections.get(command, {}).get(name) or config.shared.get(name)
    if setting is not None and not isinstance(setting.value, str):
        raise ConfigError(f"{setting.source}: {name} should be a string.")
    return setting


class _Fields(dict):
    """What ``{...}`` in a value may name, refusing anything else or unset."""

    def __init__(self, *, dest: str, source: Path, **values: str | None) -> None:
        super().__init__(values)
        self.dest = dest
        self.source = source

    def __getitem__(self, key: str) -> str:
        if key not in self:
            raise ConfigError(
                f"{self.source}: {self.dest} names {{{key}}}; only {{mode}} and "
                "{scenario} are filled in. Write {{ and }} for literal braces."
            )
        value = super().__getitem__(key)
        if value is None:
            raise ConfigError(
                f"{self.source}: {self.dest} names {{{key}}}, but no {key} is set."
            )
        return str(value)


def _fill(value: Any, fields: _Fields) -> Any:
    if isinstance(value, list):
        return [_fill(item, fields) for item in value]
    if isinstance(value, (str, Path)):
        try:
            filled = str(value).format_map(fields)
        except ConfigError:
            raise  # already says what is wrong, and where
        except (ValueError, IndexError, AttributeError) as error:
            raise ConfigError(
                f"{fields.source}: {fields.dest}: {error}. Write {{{{ and }}}} for "
                "literal braces."
            ) from None
        return Path(filled) if isinstance(value, Path) else filled
    return value


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def dump(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    command: str,
    *,
    sources: dict[str, str] | None = None,
    header: Iterable[str] = (),
) -> str:
    """
    The settings ``args`` holds for ``command``, as a file ``-c`` reads back.

    Options more than one command takes go at the top level, so the same file
    also serves the commands that share them -- ``valma view -c`` on an
    ``assign`` run's record opens it with that run's inputs. The rest go in
    the command's own section. Paths are written absolute, so the file means
    the same wherever it is moved to.
    """
    commands = subparsers(parser)
    sharing = Counter(dest for sub in commands.values() for dest in options(sub))
    top: list[str] = []
    own: list[str] = []
    for dest in options(commands[command]):
        if dest in _NOT_RECORDED:
            continue
        value = getattr(args, dest, None)
        if value is None:
            continue
        line = f"{dest} = {_toml(value)}"
        if sources is not None:
            line += f"  # {sources.get(dest, 'default')}"
        shared = sharing[dest] > 1 and dest not in _NEVER_SHARED
        (top if shared else own).append(line)

    lines = [f"# {line}" if line else "#" for line in header]
    if lines:
        lines.append("")
    lines += top
    if own:
        lines += ["", f"[{command}]", *own]
    return "\n".join(lines) + "\n"


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, Path):
        return json.dumps(Path(os.path.abspath(value)).as_posix(), ensure_ascii=False)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    if type(value) is object:
        # The sentinel a bare flag like `--omx` stands for: `omx = true`.
        return "true"
    raise TypeError(f"Cannot write {value!r} as TOML")


def sources_of(
    args: argparse.Namespace, applied: dict[str, Setting], given: set[str]
) -> dict[str, str]:
    """Where each setting came from, for ``--show-config``."""
    where: dict[str, str] = {}
    for dest in vars(args):
        if dest in given:
            where[dest] = "command line"
        elif dest in applied:
            source = applied[dest].source
            try:
                source = source.relative_to(Path.cwd())
            except ValueError:
                pass
            where[dest] = source.as_posix()
    return where

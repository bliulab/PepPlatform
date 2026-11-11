"""Language and reversible input state for the CLI; no scientific dependencies.

Collectors must be read-only. Going back replays accepted inputs against a fresh
configuration, then asks again from the preceding field. Execution belongs after
``collect`` returns, so revisiting a field cannot train a model or create a run.
"""

from __future__ import annotations

import math
import re
import shlex
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from .cli_messages import translate

TASKS = ("pep_function", "pep_hlai", "pep_hlaii", "pep_protein_interaction")


class WizardCancelled(Exception):
    """Leave the current workflow without reporting a training failure."""


class _Back(Exception):
    def __init__(self, index):
        self.index = index


@dataclass
class _Answer:
    key: tuple
    value: object
    entered: object


def field_title(title):
    title = re.sub(r"^Step [\d.]+: ", "", str(title))
    return re.sub(r"^Choose ", "", title, flags=re.IGNORECASE)


def path_value(value, *, base=None, exists=None):
    """Understand a quoted single path; keep path handling out of translations."""
    value = str(value).strip()
    if value[:1] in {"'", '"'}:
        parts = shlex.split(value)
        if len(parts) != 1:
            raise ValueError("Please enter one path.")
        value = parts[0]
    path = Path(value).expanduser()
    if not value:
        raise ValueError("A path is required.")
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    path = path.resolve()
    if exists == "file" and not path.is_file():
        raise ValueError(f"File does not exist: {path}")
    if exists == "directory" and not path.is_dir():
        raise ValueError(f"Directory does not exist: {path}")
    return path


class Interaction:
    def __init__(self, language=None, *, steps=True):
        self.language = language
        self.steps = steps
        self.answers = []
        self.retained_defaults = {}
        self.cursor = 0
        self.replay_until = 0
        self.started = False

    def t(self, message):
        return translate(str(message), self.language or "zh")

    def say(self, message=""):
        if self.cursor >= self.replay_until:
            print(self.t(message))

    def choose_language(self):
        self.started = True
        if self.language is not None:
            return self.language
        print("\nLanguage / 语言\n  1. 中文\n  2. English")
        while True:
            raw = input("Choice / 选择 [1] (q: cancel / 取消): ").strip().lower()
            if raw in {"q", "b"}:
                raise WizardCancelled()
            if raw in {"", "1", "zh", "中文"}:
                self.language = "zh"
                return self.language
            if raw in {"2", "en", "english"}:
                self.language = "en"
                return self.language
            print("Choose 1 or 2 / 请输入 1 或 2。")

    def collect(self, callback):
        """Replay only input collection, never the operation being configured."""
        while True:
            self.cursor = 0
            try:
                return callback()
            except _Back as back:
                if back.index < 0:
                    raise WizardCancelled() from None
                self.replay_until = back.index

    def rewind(self, index=0):
        raise _Back(index)

    def retry_field(self, title):
        title = field_title(title)
        for index, answer in enumerate(self.answers):
            if answer.key[1] == title:
                self.rewind(index)
        self.rewind()

    def _ask(self, title, default, parse, *, kind="text", options=()):
        title = field_title(title)
        key = (kind, title, tuple(options))
        index = self.cursor
        previous = self.answers[index] if index < len(self.answers) else None
        if previous is not None and previous.key != key:
            for answer in self.answers[index:]:
                self.retained_defaults[answer.key] = answer
            del self.answers[index:]
            previous = None
            self.replay_until = min(self.replay_until, index)
        if index < self.replay_until and previous is not None:
            self.cursor += 1
            return deepcopy(previous.value)
        if previous is not None:
            default = deepcopy(previous.entered)
        elif key in self.retained_defaults:
            default = deepcopy(self.retained_defaults[key].entered)
        prefix = f"Step {index + 1}: " if self.steps else ""
        print("\n" + self.t(prefix + title))
        if options:
            for number, value in enumerate(options, 1):
                print(f"  {number}. {self.t(value)}")
            display_default = options.index(default) + 1 if default in options else 1
        else:
            display_default = "" if default is None else default
        while True:
            prompt = self.t("Choice" if options else "Value")
            controls = (
                "(b: back, q: cancel, -: clear): " if kind == "text" else "(b: back, q: cancel): "
            )
            prompt += f" [{display_default}] " + self.t(controls)
            raw = input(prompt).strip()
            if raw.lower() in {"b", "__back__"}:
                raise _Back(index - 1)
            if raw.lower() == "q":
                raise WizardCancelled()
            try:
                value = parse(raw, default)
            except (ValueError, OSError) as exc:
                self.say(str(exc))
                continue
            answer = _Answer(
                key,
                deepcopy(value),
                ("" if raw == "-" else deepcopy(raw or default))
                if kind == "text"
                else deepcopy(value),
            )
            if index < len(self.answers):
                self.answers[index] = answer
            else:
                self.answers.append(answer)
            self.cursor += 1
            return value

    def choice(self, title, options, default=None):
        options = list(options)
        if default not in options:
            default = options[0]

        def parse(raw, current):
            if not raw:
                return current
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1]
            for value in options:
                if raw.casefold() in {str(value).casefold(), self.t(value).casefold()}:
                    return value
            raise ValueError("Choose a listed number.")

        return self._ask(title, default, parse, kind="choice", options=options)

    def integer(self, title, default, minimum=None, maximum=None):
        def parse(raw, current):
            try:
                value = int(raw) if raw else current
                if value is None:
                    raise ValueError()
            except (TypeError, ValueError):
                raise ValueError("Please enter an integer.") from None
            if minimum is not None and value < minimum:
                raise ValueError(f"Please enter an integer >= {minimum}.")
            if maximum is not None and value > maximum:
                raise ValueError(f"Please enter an integer <= {maximum}.")
            return value

        return self._ask(title, default, parse, kind="integer")

    def number(self, title, default, minimum=None, *, exclusive_minimum=False):
        def parse(raw, current):
            try:
                value = float(raw) if raw else float(current)
            except (TypeError, ValueError):
                raise ValueError("Please enter a number.") from None
            if not math.isfinite(value):
                raise ValueError("Please enter a finite number.")
            if minimum is not None and (value <= minimum if exclusive_minimum else value < minimum):
                operator = ">" if exclusive_minimum else ">="
                raise ValueError(f"Please enter a number {operator} {minimum}.")
            return value

        return self._ask(title, default, parse, kind="number")

    def text(self, title, default="", *, validate=None):
        def parse(raw, current):
            value = "" if raw == "-" else raw or current or ""
            return validate(value) if validate is not None else value

        return self._ask(title, default, parse)

    def confirm(self):
        action = self.choice("Review configuration", ["Run", "Modify", "Cancel"], "Modify")
        if action == "Cancel":
            raise WizardCancelled()
        if action == "Modify":
            self.rewind()

    def result(self, result):
        """Show a human result while preserving the operation's JSON payload."""
        import json

        self.say("Operation complete")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

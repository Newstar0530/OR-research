"""Turn a subprocess traceback into something a repair step can act on.

A failed experiment currently leaves behind a wall of stderr. Most of it is
noise -- frames from this project's own machinery, absolute paths that exist
only on one machine, library internals -- and the three facts that matter (what
was raised, where, and what the line said) are buried in it.

That noise is not merely untidy. Whatever consumes this next, a deterministic
rule or a language model, will spend its attention on whichever frames it sees,
and frames belonging to the framework invite "fixes" to the framework rather
than to the generated experiment. So the parser keeps user frames, drops the
rest, and rewrites absolute workspace paths back to the script's own name.

Nothing here guesses. When stderr contains no traceback -- a timeout, a bare
non-zero exit -- the result says so rather than inventing an exception.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, Field


#: Path fragments that mark a frame as belonging to the machinery rather than to
#: the generated experiment.
FRAMEWORK_MARKERS: tuple[str, ...] = (
    "/src/core/",
    "\\src\\core\\",
    "/src/execution/",
    "\\src\\execution\\",
    "/src/agents/",
    "\\src\\agents\\",
    "/src/utils/",
    "\\src\\utils\\",
    "site-packages",
    "dist-packages",
    "importlib",
    "<frozen ",
    "runpy.py",
)

_FRAME_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<name>.+))?\s*$')
_EXCEPTION_RE = re.compile(r"^(?P<type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Exit|Warning|Interrupt))(?:: (?P<message>.*))?$")
_TRACEBACK_HEADER = "Traceback (most recent call last):"


class TracebackFrame(BaseModel):
    filename: str
    lineno: int
    name: str = ""
    line: str = ""
    is_user_code: bool = False

    def render(self) -> str:
        location = f'  File "{self.filename}", line {self.lineno}'
        if self.name:
            location += f", in {self.name}"
        return location + (f"\n    {self.line}" if self.line else "")


class ParsedTraceback(BaseModel):
    """What actually went wrong, with the framework's own frames removed."""

    exception_type: str | None = None
    exception_message: str | None = None
    frames: list[TracebackFrame] = Field(default_factory=list)
    is_syntax_error: bool = False
    filtered_text: str = ""
    raw_tail: str = ""
    parse_note: str = ""

    @property
    def has_exception(self) -> bool:
        return bool(self.exception_type)

    @property
    def user_frames(self) -> list[TracebackFrame]:
        return [frame for frame in self.frames if frame.is_user_code]

    @property
    def failing_frame(self) -> TracebackFrame | None:
        """The deepest frame in the generated code, or the deepest frame at all."""

        user = self.user_frames
        if user:
            return user[-1]
        return self.frames[-1] if self.frames else None

    def summary(self) -> str:
        if not self.has_exception:
            return self.parse_note or "no exception was raised"
        frame = self.failing_frame
        where = f" at {frame.filename}:{frame.lineno}" if frame else ""
        message = f": {self.exception_message}" if self.exception_message else ""
        return f"{self.exception_type}{message}{where}"


def _shorten(path_text: str, work_dir: Path | None, script_name: str) -> str:
    """`/long/node/dir/experiment.py` -> `experiment.py`."""

    if work_dir is not None:
        for candidate in {str(work_dir), str(work_dir.resolve())}:
            if path_text.startswith(candidate):
                return path_text[len(candidate) :].lstrip("/\\") or script_name
    return path_text


def _is_user_frame(filename: str, work_dir: Path | None, script_name: str) -> bool:
    if any(marker in filename for marker in FRAMEWORK_MARKERS):
        return False
    if Path(filename).name == script_name:
        return True
    if work_dir is not None:
        try:
            Path(filename).resolve().relative_to(work_dir.resolve())
            return True
        except (ValueError, OSError):
            return False
    return False


def parse_traceback(
    stderr: str,
    work_dir: str | Path | None = None,
    script_name: str = "experiment.py",
) -> ParsedTraceback:
    """Extract the last traceback from `stderr`, keeping only user frames."""

    text = stderr or ""
    root = Path(work_dir) if work_dir is not None else None
    if _TRACEBACK_HEADER not in text:
        stripped = text.strip()
        if not stripped:
            return ParsedTraceback(parse_note="stderr was empty")
        # A bare `SyntaxError: ...` or a message with no traceback still names a failure.
        for line in reversed(stripped.splitlines()):
            match = _EXCEPTION_RE.match(line.strip())
            if match:
                return ParsedTraceback(
                    exception_type=match.group("type"),
                    exception_message=(match.group("message") or "").strip() or None,
                    is_syntax_error=match.group("type") == "SyntaxError",
                    filtered_text=stripped[-2000:],
                    raw_tail=stripped[-2000:],
                    parse_note="no traceback block; recovered the exception line only",
                )
        return ParsedTraceback(
            parse_note="stderr contained no traceback and no recognisable exception",
            raw_tail=stripped[-2000:],
        )

    block = text[text.rindex(_TRACEBACK_HEADER) :]
    lines = block.splitlines()
    frames: list[TracebackFrame] = []
    index = 1
    while index < len(lines):
        match = _FRAME_RE.match(lines[index])
        if not match:
            break
        filename = match.group("file")
        source = ""
        if index + 1 < len(lines) and not _FRAME_RE.match(lines[index + 1]):
            candidate = lines[index + 1]
            if candidate.startswith("    ") and candidate.strip() and not candidate.strip().startswith("^"):
                source = candidate.strip()
                index += 1
        frames.append(
            TracebackFrame(
                filename=_shorten(filename, root, script_name),
                lineno=int(match.group("line")),
                name=(match.group("name") or "").strip(),
                line=source,
                is_user_code=_is_user_frame(filename, root, script_name),
            )
        )
        index += 1

    exception_type = exception_message = None
    for line in reversed(lines[index:] or lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("^") or stripped.startswith("~"):
            continue
        match = _EXCEPTION_RE.match(stripped)
        if match:
            exception_type = match.group("type")
            exception_message = (match.group("message") or "").strip() or None
            break

    kept = [frame for frame in frames if frame.is_user_code] or frames
    rendered = [_TRACEBACK_HEADER] + [frame.render() for frame in kept]
    if exception_type:
        rendered.append(
            f"{exception_type}: {exception_message}" if exception_message else exception_type
        )
    dropped = len(frames) - len(kept)
    note = ""
    if dropped > 0:
        note = f"{dropped} framework frame(s) removed so the failure points at the experiment"

    return ParsedTraceback(
        exception_type=exception_type,
        exception_message=exception_message,
        frames=frames,
        is_syntax_error=exception_type == "SyntaxError",
        filtered_text="\n".join(rendered),
        raw_tail=block[-2000:],
        parse_note=note,
    )


def code_excerpt(
    path: str | Path, lineno: int, context: int = 4, max_line_length: int = 200
) -> str:
    """The failing line with a few lines either side, numbered and marked."""

    source = Path(path)
    if not source.exists() or lineno < 1:
        return ""
    try:
        lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:  # pragma: no cover - defensive
        return ""
    start = max(0, lineno - 1 - context)
    stop = min(len(lines), lineno + context)
    width = len(str(stop))
    rendered = []
    for number in range(start, stop):
        marker = ">>" if number == lineno - 1 else "  "
        body = lines[number][:max_line_length]
        rendered.append(f"{marker} {str(number + 1).rjust(width)} | {body}")
    return "\n".join(rendered)


class RepairContext(BaseModel):
    """Everything a repair step should see, and nothing it should not."""

    script_name: str = "experiment.py"
    exception_type: str | None = None
    exception_message: str | None = None
    failing_file: str | None = None
    failing_line: int | None = None
    failing_source: str | None = None
    code_excerpt: str = ""
    filtered_traceback: str = ""
    stdout_tail: str = ""
    attempt: int = 0
    previous_attempts: list[str] = Field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return bool(self.exception_type)

    def to_prompt(self, max_stdout: int = 800) -> str:
        sections = [
            f"The generated experiment `{self.script_name}` failed.",
            "",
            f"Exception: {self.exception_type or 'unknown'}"
            + (f": {self.exception_message}" if self.exception_message else ""),
        ]
        if self.failing_file and self.failing_line:
            sections.append(f"Location: {self.failing_file}, line {self.failing_line}")
        if self.code_excerpt:
            sections.extend(["", "Code around the failure:", "```python", self.code_excerpt, "```"])
        if self.filtered_traceback:
            sections.extend(["", "Traceback (framework frames removed):", "```", self.filtered_traceback, "```"])
        if self.stdout_tail:
            sections.extend(["", "Last stdout before the failure:", "```", self.stdout_tail[-max_stdout:], "```"])
        if self.previous_attempts:
            sections.extend(["", "Repairs already tried on this node (do not repeat them):"])
            sections.extend(f"- {item}" for item in self.previous_attempts)
        sections.extend(
            [
                "",
                "Change only the smallest thing that fixes this error. Preserve the "
                "results.csv output and the SUMMARY_JSON stdout line.",
            ]
        )
        return "\n".join(sections)


def build_repair_context(
    stderr: str,
    work_dir: str | Path,
    script_name: str = "experiment.py",
    stdout: str = "",
    attempt: int = 0,
    previous_attempts: Sequence[str] = (),
    context_lines: int = 4,
) -> RepairContext:
    """Parse a failure and gather the code around it."""

    root = Path(work_dir)
    parsed = parse_traceback(stderr, work_dir=root, script_name=script_name)
    frame = parsed.failing_frame
    excerpt = ""
    failing_file = failing_line = None
    if frame is not None:
        failing_file = frame.filename
        failing_line = frame.lineno
        candidate = root / Path(frame.filename).name
        if candidate.exists():
            excerpt = code_excerpt(candidate, frame.lineno, context=context_lines)
    return RepairContext(
        script_name=script_name,
        exception_type=parsed.exception_type,
        exception_message=parsed.exception_message,
        failing_file=failing_file,
        failing_line=failing_line,
        failing_source=frame.line if frame else None,
        code_excerpt=excerpt,
        filtered_traceback=parsed.filtered_text,
        stdout_tail=(stdout or "")[-2000:],
        attempt=attempt,
        previous_attempts=list(previous_attempts),
    )

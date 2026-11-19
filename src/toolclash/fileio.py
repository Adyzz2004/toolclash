"""Reading and writing toolclash's JSON files.

Always UTF-8, whatever the platform's default encoding is. On Windows the default
is a legacy code page: reading UTF-8 tool files with it garbles any non-English
text before the model sees it, and writing with it crashes on characters such as
"≥", "→" or emoji that a generator model may well produce.
"""

from __future__ import annotations

import locale
import os
import secrets
import stat
import sys
import time
from pathlib import Path

# Windows tools add byte-order marks: PowerShell 5's `>` writes UTF-16, Out-File and
# Notepad may write UTF-8 with a BOM. JSON parsers reject the BOM, so strip it here.
_BOMS = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
)


class EncodingError(ValueError):
    """A file is not UTF-8 (or UTF-16 with a byte-order mark)."""


def read_text(path: str | Path, *, legacy: bool = False) -> str:
    """Read a text file as UTF-8 (UTF-16 too, when it starts with a byte-order mark).

    ``legacy=True`` is for files toolclash itself wrote: earlier versions used the
    platform's default encoding (a Windows code page), so such a file is read with
    that encoding when it is not valid UTF-8. Tool files never get this fallback: a
    wrong guess there would silently garble what the model is shown.
    """
    data = Path(path).read_bytes()
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            return data.decode(encoding)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        if legacy:
            try:
                return data.decode(_legacy_encoding())
            except (UnicodeDecodeError, LookupError):
                pass
        raise EncodingError(
            f"{path} is not UTF-8 text ({e.reason} at byte {e.start}); save it as UTF-8 and try again"
        ) from e


def _legacy_encoding() -> str:
    """The encoding earlier versions wrote with: the locale's, even in Python's UTF-8 mode."""
    getencoding = getattr(locale, "getencoding", None)  # Python 3.11+; ignores UTF-8 mode
    if getencoding:
        return getencoding()
    if sys.platform == "win32":  # Python 3.10: ask Windows for its ANSI code page directly
        import ctypes

        return f"cp{ctypes.windll.kernel32.GetACP()}"
    return locale.getpreferredencoding(False)


def _on_windows() -> bool:
    return sys.platform == "win32"


_BAD_WINDOWS_NAME_CHARS = set('<>:"|?*')


def check_file_name(path: str | Path) -> None:
    """Reject a name Windows cannot store (``a:b`` would silently become a hidden stream)."""
    name = Path(path).name
    if _on_windows() and _BAD_WINDOWS_NAME_CHARS & set(name):
        raise ValueError(f'{name} is not a valid file name on Windows (no <>:"|?* allowed)')


def write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text, creating parent folders as needed.

    The file is written to a temporary name first and then moved into place, so a
    failure never leaves a half-written file or destroys the previous version.
    """
    path = Path(path)
    if path.is_dir():
        raise IsADirectoryError(f"{path} is a folder, not a file")
    check_file_name(path)
    try:
        existing = os.stat(path)
    except OSError:
        existing = None
    if existing is not None and not stat.S_ISREG(existing.st_mode):
        # A device or pipe (NUL, CON, /dev/null, /dev/stdout): there is nothing to replace.
        _write_in_place(path, text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd, tmp = _create_temp(path)
    except PermissionError:
        if existing is None:
            raise
        _write_in_place(path, text)  # the folder lets us change the file but not add new ones
        return
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        if existing is not None:  # keep the mode the file had
            os.chmod(tmp, stat.S_IMODE(existing.st_mode))
        _replace(tmp, path)
    except BaseException:
        _discard(tmp)
        raise


def _write_in_place(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _create_temp(path: Path) -> tuple[int, str]:
    """A new empty file next to ``path``, created with the usual umask-based mode.

    (tempfile.mkstemp would make it owner-only, and changing the umask to work out
    the usual mode is not safe while other threads create files.)
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        tmp = str(path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp"))
        try:
            return os.open(tmp, flags, 0o666), tmp
        except FileExistsError:
            continue
    raise FileExistsError(f"could not create a temporary file next to {path}")


def _discard(tmp: str) -> None:
    """Remove a temporary file, even a read-only one, without masking the real error."""
    try:
        os.chmod(tmp, 0o666)
        os.unlink(tmp)
    except OSError:
        pass


def _replace(tmp: str, path: Path) -> None:
    """os.replace, made to work when another program has the file open on Windows.

    Windows refuses to replace a file that someone else holds open (a viewer, an
    indexer, a log tailer) where a plain overwrite would still succeed. Retry briefly,
    then fall back to overwriting the file in place.
    """
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if not _on_windows():
                raise
            time.sleep(0.05 * (attempt + 1))
    try:
        _write_in_place(path, Path(tmp).read_text(encoding="utf-8"))
    except PermissionError as e:
        raise PermissionError(f"cannot write {path}: it is read-only or open in another program") from e
    _discard(tmp)

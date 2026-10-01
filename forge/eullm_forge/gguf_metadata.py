"""GGUF metadata, read and written with the standard library alone.

A GGUF from llama.cpp's converter carries what the converter knows. What
Forge learnt while training — the temperature a decision model is
calibrated at, say — is added afterwards: `set_metadata` adds, replaces or
removes keys of a GGUF that exists, and `read_metadata` reads them back,
for Forge and for bench/reflexbench/qualify.py, which needs nothing beyond
the standard library either.

A key is added by writing the file again: the header with the new count of
key-value pairs, every pair already there byte for byte, the new one, every
tensor's description byte for byte, then the tensor data as it was. Nothing
here interprets a tensor, so a GGUF of any quantization, from any
llama.cpp, comes through unchanged; only the padding before the data,
which the format aligns to `general.alignment` (32 bytes unless it says
otherwise), is written anew, since the header's length changed. The new
file is written beside the old one and renamed over it: a failure halfway
leaves the GGUF as it was.

GGUF versions 2 and 3, little-endian — what llama.cpp writes and reads
(ggml/src/gguf.cpp). Version 1, which llama.cpp no longer reads, and a
big-endian file are refused rather than misread.
"""

from __future__ import annotations

import os
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"GGUF"
VERSIONS = (2, 3)
#: The magic, the version, and the counts of tensors and of key-value pairs.
_HEADER = struct.Struct("<4sIQQ")
#: gguf.h `GGUF_DEFAULT_ALIGNMENT`, and the key that changes it.
DEFAULT_ALIGNMENT = 32
ALIGNMENT_KEY = "general.alignment"

#: gguf.h `enum gguf_type`: name → (id, struct format). A string, and an
#: array, have no fixed size: their length comes first.
TYPES = {
    "uint8": (0, "<B"),
    "int8": (1, "<b"),
    "uint16": (2, "<H"),
    "int16": (3, "<h"),
    "uint32": (4, "<I"),
    "int32": (5, "<i"),
    "float32": (6, "<f"),
    "bool": (7, "<?"),
    "string": (8, None),
    "array": (9, None),
    "uint64": (10, "<Q"),
    "int64": (11, "<q"),
    "float64": (12, "<d"),
}
_NAMES = {type_id: name for name, (type_id, _) in TYPES.items()}


class GGUFError(ValueError):
    """A file that is not a GGUF this module reads, or a value it cannot
    write."""


@dataclass
class _Layout:
    """Where the parts of a GGUF's header are, as offsets in the file."""

    version: int
    n_tensors: int
    #: Every key-value pair in file order: (key, start, end).
    pairs: list = field(default_factory=list)
    #: The tensors' descriptions: (start, end).
    infos: tuple = (0, 0)
    alignment: int = DEFAULT_ALIGNMENT
    #: Where the tensor data starts.
    data: int = 0


class _Reader:
    """A GGUF's header, read within the file's size: a length that runs
    past the end is a damaged file, not something to allocate."""

    def __init__(self, f, size: int):
        self.f, self.size = f, size

    def take(self, n: int) -> bytes:
        if n < 0 or self.f.tell() + n > self.size:
            raise GGUFError("the header runs past the end of the file")
        return self.f.read(n)

    def skip(self, n: int) -> None:
        if n < 0 or self.f.tell() + n > self.size:
            raise GGUFError("the header runs past the end of the file")
        self.f.seek(n, 1)

    def number(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self, read: bool = True) -> str | None:
        n = self.number("<Q")
        if not read:
            self.skip(n)
            return None
        return self.take(n).decode("utf-8", errors="replace")

    def value(self, type_id: int, read: bool):
        """The value of `type_id` at the current offset, or None, having
        skipped it, when not `read`. An array is a list."""
        name = _NAMES.get(type_id)
        if name is None:
            raise GGUFError(f"unknown value type {type_id}")
        if name == "string":
            return self.string(read)
        if name == "array":
            item, count = self.number("<I"), self.number("<Q")
            item_name = _NAMES.get(item)
            if item_name is None or item_name == "array":
                # gguf.cpp refuses an array of arrays too.
                raise GGUFError(f"an array of value type {item} is not one llama.cpp reads")
            fmt = TYPES[item_name][1]
            if fmt and not read:
                self.skip(count * struct.calcsize(fmt))
                return None
            values = []
            for _ in range(count):
                values.append(self.value(item, read))
            return values if read else None
        fmt = TYPES[name][1]
        if not read:
            self.skip(struct.calcsize(fmt))
            return None
        return self.number(fmt)


def _pad(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _parse(path: Path, wanted: set | None) -> tuple[_Layout, dict]:
    """The layout of the GGUF at `path`, and `(type, value)` of the keys
    `wanted` — every key when None. `general.alignment` is always read."""
    size = path.stat().st_size
    values: dict = {}
    with open(path, "rb", buffering=1 << 20) as f:
        r = _Reader(f, size)
        if size < _HEADER.size or r.take(4) != MAGIC:
            raise GGUFError(f"{path} is not a GGUF file")
        version = r.number("<I")
        if version and not version & 0xFFFF:
            raise GGUFError(f"{path} is a big-endian GGUF, which this does not read")
        if version not in VERSIONS:
            raise GGUFError(f"{path} is GGUF version {version}; this reads versions 2 and 3")
        layout = _Layout(version, r.number("<Q"))
        n_pairs = r.number("<Q")
        seen = set()
        for _ in range(n_pairs):
            start = f.tell()
            try:
                key = r.take(r.number("<Q")).decode("utf-8")
            except UnicodeDecodeError:
                raise GGUFError(f"{path}: a key that is not UTF-8") from None
            if not key or key in seen:
                # gguf.cpp refuses both: the file is damaged.
                raise GGUFError(f"{path}: an empty or repeated key {key!r}")
            seen.add(key)
            type_id = r.number("<I")
            read = wanted is None or key in wanted or key == ALIGNMENT_KEY
            value = r.value(type_id, read)
            if read:
                values[key] = (_NAMES[type_id], value)
            layout.pairs.append((key, start, f.tell()))
        if ALIGNMENT_KEY in values:
            kind, alignment = values[ALIGNMENT_KEY]
            if kind != "uint32" or alignment <= 0 or alignment & (alignment - 1):
                raise GGUFError(f"{path}: {ALIGNMENT_KEY} must be a power of 2, a uint32")
            layout.alignment = alignment
        start = f.tell()
        for _ in range(layout.n_tensors):
            r.string(read=False)
            r.skip(8 * r.number("<I"))  # the shape, one uint64 a dimension
            r.skip(4 + 8)  # the type, and the offset in the data
        end = f.tell()
        layout.infos = (start, end)
        # gguf.cpp seeks to the aligned data only when there are tensors.
        layout.data = _pad(end, layout.alignment) if layout.n_tensors else end
        if layout.data > size:
            raise GGUFError(f"{path} ends before its tensor data")
    return layout, values


def read_fields(path: str | Path, keys=None) -> dict:
    """The metadata of the GGUF at `path`, key → `(type, value)` such as
    `("float32", 1.37)`; an array's value is a list. With `keys`, only
    those, every other value skipped unread."""
    wanted = None if keys is None else set(keys)
    _, values = _parse(Path(path), wanted)
    if wanted is not None:
        values = {k: v for k, v in values.items() if k in wanted}
    return values


def read_metadata(path: str | Path, keys=None) -> dict:
    """The metadata of the GGUF at `path`, key → value; with `keys`, only
    those that are there."""
    return {key: value for key, (_, value) in read_fields(path, keys).items()}


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def encode(type_name: str, value) -> bytes:
    """A value as a key-value pair holds it: its type, then the value.
    Raises GGUFError for a type or a value a GGUF cannot hold so."""
    if type_name not in TYPES or type_name == "array":
        names = ", ".join(t for t in TYPES if t != "array")
        raise GGUFError(f"a metadata value is one of {names}, not {type_name!r}")
    type_id, fmt = TYPES[type_name]
    if type_name == "string":
        if not isinstance(value, str):
            raise GGUFError(f"a string, not {value!r}")
        return struct.pack("<I", type_id) + _string(value)
    if type_name == "bool":
        ok = isinstance(value, bool)
    elif type_name.startswith("float"):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        ok = isinstance(value, int) and not isinstance(value, bool)
    if not ok:
        raise GGUFError(f"{value!r} is not a {type_name}")
    try:
        return struct.pack("<I", type_id) + struct.pack(fmt, value)
    except (struct.error, OverflowError) as e:
        raise GGUFError(f"{value!r} does not fit a {type_name}: {e}") from e


def encode_updates(updates: dict) -> dict:
    """`set_metadata`'s updates as the bytes of their pairs, or None for a
    key to remove: what it will write, checked before anything is."""
    encoded = {}
    for key, update in updates.items():
        if not isinstance(key, str) or not key:
            raise GGUFError(f"a metadata key is a non-empty string, not {key!r}")
        if key == ALIGNMENT_KEY:
            raise GGUFError(f"{ALIGNMENT_KEY} moves every tensor: it is not changed here")
        if update is None:
            encoded[key] = None
            continue
        if not isinstance(update, (tuple, list)) or len(update) != 2:
            raise GGUFError(f"{key}: a value is (type, value), such as ('float32', 1.5)")
        encoded[key] = _string(key) + encode(*update)
    return encoded


def set_metadata(path: str | Path, updates: dict) -> bool:
    """Add or replace metadata of the GGUF at `path` — key → `(type,
    value)`, such as `("float32", 1.37)` — or remove a key given as None.
    A key already there keeps its place, a new one goes last. Returns
    whether the file changed: when nothing would, it is not written."""
    path = Path(path)
    encoded = encode_updates(updates)
    layout, _ = _parse(path, wanted=set())
    with open(path, "rb") as f:
        header = f.read(layout.infos[1])
    pairs = []
    for key, start, end in layout.pairs:
        if key not in encoded:
            pairs.append(header[start:end])
        elif encoded[key] is not None:
            pairs.append(encoded[key])
    present = {key for key, _, _ in layout.pairs}
    pairs += [pair for key, pair in encoded.items() if key not in present and pair is not None]
    kv = b"".join(pairs)
    if kv == header[_HEADER.size:layout.infos[0]]:
        return False
    head = (_HEADER.pack(MAGIC, layout.version, layout.n_tensors, len(pairs)) + kv
            + header[layout.infos[0]:layout.infos[1]])
    if layout.n_tensors:
        head += b"\0" * (_pad(len(head), layout.alignment) - len(head))
    partial = path.with_name(path.name + ".partial")
    try:
        with open(path, "rb") as src, open(partial, "wb") as dst:
            dst.write(head)
            src.seek(layout.data)
            shutil.copyfileobj(src, dst, 16 << 20)
        shutil.copymode(path, partial)
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return True

"""Read sampled skeletal tracks from the Prinny NMA/NMBA animation files."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .nmba_facial import FacialTrack, read_facial_tracks
    from .nmbm_to_obj import chunks
    from .ykcmp import MAGIC, decompress
else:
    from nmba_facial import FacialTrack, read_facial_tracks
    from nmbm_to_obj import chunks
    from ykcmp import MAGIC, decompress


# Matches the Maya rotateOrder enumeration used by these exported rigs.
ROTATION_ORDERS = ("XYZ", "YZX", "ZXY", "XZY", "YXZ", "ZYX")


@dataclass
class Track:
    identifier: int
    name: str
    path: str
    rotation_order: str
    channels: dict[int, tuple[tuple[float, float, float], ...]]

    def value(self, semantic: int, sample: int):
        values = self.channels[semantic]
        return values[0 if len(values) == 1 else sample]


@dataclass
class Animation:
    name: str
    fps: float
    times: tuple[float, ...]
    tracks: tuple[Track, ...]
    extra_chunks: tuple[str, ...]
    facial_tracks: tuple[FacialTrack, ...] = ()


def load_animation(source: Path) -> Animation:
    data = source.read_bytes()
    if data[:4] != b"NMA\0":
        offset = data.find(MAGIC)
        if offset < 0:
            raise ValueError("input is neither NMA nor YKCMP-wrapped NMBA")
        data = decompress(data, offset)
    return parse_animation(data, source.stem)


def parse_animation(data: bytes, name: str) -> Animation:
    if data[:4] != b"NMA\0":
        raise ValueError("animation does not start with NMA magic")
    sections = list(chunks(data))

    def section(magic):
        found = [item for item in sections if item[0] == magic]
        if len(found) != 1:
            raise ValueError(f"expected one {magic!r} animation chunk")
        _, offset, header, size = found[0]
        if not 0x20 <= header <= size:
            raise ValueError("invalid animation chunk header")
        return offset + header, offset + size

    def unpack(fmt, offset, end):
        if offset < 0 or offset + struct.calcsize(fmt) > end:
            raise ValueError(f"truncated animation data at {offset:#x}")
        return struct.unpack_from(fmt, data, offset)

    base, end = section(b"TIME")
    start, stop, _loop_start, _loop_stop, fps, count = unpack("<5fI", base, end)
    if not math.isfinite(fps) or fps <= 0 or count < 1:
        raise ValueError("invalid animation timing")
    times = unpack(f"<{count}f", base + 0x20, end)
    if (not all(math.isfinite(t) for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))
            or abs(times[0] - start) > 1e-5 or abs(times[-1] - stop) > 1e-5):
        raise ValueError("unsupported or inconsistent animation timeline")

    base, end = section(b"TRSA")
    count = unpack("<I", base + 0x14, end)[0]
    cursor = base + 0x1C
    # Names follow the record area. The last record's next offset is zero.
    records_end = cursor + unpack("<I", cursor + 0x10, end)[0]
    if count < 1 or not cursor + 32 <= records_end < end:
        raise ValueError("invalid TRSA record area")
    tracks = []
    paths = set()
    for record_index in range(count):
        identifier, order = unpack("<HH", cursor, end)
        fields = unpack("<8I", cursor, end)
        record_end = cursor + fields[7] if fields[7] else records_end
        if (not cursor + 32 <= record_end <= records_end
                or (fields[7] == 0) != (record_index == count - 1)
                or order >= len(ROTATION_ORDERS)):
            raise ValueError("invalid TRSA record or rotation order")

        def text_at(relative):
            offset = cursor + relative
            if not base <= offset < end:
                raise ValueError("TRSA name offset is out of range")
            terminator = data.find(b"\0", offset, end)
            if terminator < 0:
                raise ValueError("unterminated TRSA name")
            return data[offset:terminator].decode("utf-8")

        short_name, path = text_at(fields[4]), text_at(fields[5])
        if not short_name or path.split("|")[-1] != short_name or path in paths:
            raise ValueError("invalid or duplicate TRSA node path")
        paths.add(path)
        channel_offset = cursor + 32
        channels = {}
        for semantic in range(0x0E, 0x13):
            size, actual_semantic, key_count, flags = unpack("<IHHI", channel_offset, record_end)
            channel_end = channel_offset + size if size else record_end
            if (actual_semantic != semantic or flags != 0
                    or key_count not in (1, len(times))
                    or not channel_offset + 12 + key_count * 20 <= channel_end <= record_end
                    or (size == 0) != (semantic == 0x12)):
                raise ValueError(f"unsupported TRSA channel for {path}")
            values = []
            for key in range(key_count):
                kind, x, y, z, w = unpack("<I4f", channel_offset + 12 + key * 20, channel_end)
                expected_kinds = (1,) if semantic == 0x12 else (2, 5, 10)
                if kind not in expected_kinds or w != 1 or not all(math.isfinite(v) for v in (x, y, z)):
                    raise ValueError(f"unsupported key encoding for {path}")
                values.append((x, y, z))
            channels[semantic] = tuple(values)
            channel_offset = channel_end
        if channels[0x11] != ((0.0, 0.0, 0.0),) or channels[0x12] != ((1.0, 0.0, 0.0),):
            raise ValueError(f"unsupported auxiliary TRSA channel for {path}")
        tracks.append(Track(identifier, short_name, path, ROTATION_ORDERS[order], channels))
        cursor = record_end
    facial = read_facial_tracks(data, sections, len(times))
    extra = tuple(m.decode("ascii") for m, *_ in sections if m not in (b"INFO", b"TIME", b"TRSA", b"BLSA", b"EOFC"))
    return Animation(name, fps, times, tuple(tracks), extra, facial)

"""Read sampled BLSA blend-shape weights, including repeated equivalent tracks."""

from dataclasses import dataclass
import math
import struct


@dataclass(frozen=True)
class FacialTrack:
    name: str
    values: tuple[float, ...]

    def value(self, sample):
        return self.values[0 if len(self.values) == 1 else sample]


def read_facial_tracks(data, sections, sample_count):
    found = [c for c in sections if c[0] == b"BLSA"]
    if not found:
        return ()
    if len(found) != 1:
        raise ValueError("expected at most one BLSA chunk")
    _, offset, header, size = found[0]
    base, end = offset + header, offset + size
    if not 0x20 <= header <= size or base + 28 > end:
        raise ValueError("invalid BLSA header")
    start_offset = struct.unpack_from("<I", data, base + 4)[0]
    string_offset, stride, count = struct.unpack_from("<3I", data, base + 16)
    cursor = base + start_offset
    strings = cursor + string_offset
    if start_offset != 28 or stride != 16 or not cursor <= strings < end:
        raise ValueError("unsupported BLSA table layout")
    result = {}
    for _ in range(count):
        if cursor + 16 > strings:
            raise ValueError("truncated BLSA track")
        # Only the type byte, uint16 count, and name offset are meaningful.
        # Unused header bytes contain leftover string data in shipped files.
        kind, key_count = struct.unpack_from("<BxH", data, cursor + 4)
        name_offset = struct.unpack_from("<I", data, cursor + 12)[0]
        start = strings + name_offset
        stop = data.find(b"\0", start, end) if start < end else -1
        track_end = cursor + 16 + key_count * 8
        if (kind != 1 or key_count not in (1, sample_count) or stop <= start
                or track_end > strings):
            raise ValueError("unsupported BLSA track encoding")
        name = data[start:stop].decode("utf-8")
        values = []
        for i in range(key_count):
            key = cursor + 16 + i * 8
            value = struct.unpack_from("<f", data, key + 4)[0]
            if data[key] not in (5, 11) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"unsupported BLSA scalar key: {name}")
            values.append(value)
        track = FacialTrack(name, tuple(values))
        if name.rsplit(":", 1)[-1] == "BlendShape":
            if any(v != 1 for v in values):
                raise ValueError("animated BLSA envelope is unsupported")
        elif name in result:
            # Machange clips repeat the controls below a namespaced envelope.
            # Collapse them only if every sample agrees, never choose arbitrarily.
            if any(result[name].value(i) != track.value(i) for i in range(sample_count)):
                raise ValueError(f"conflicting duplicate BLSA track: {name}")
        else:
            result[name] = track
        cursor = track_end
    if cursor != strings:
        raise ValueError("BLSA records do not reach the string table")
    return tuple(result.values())

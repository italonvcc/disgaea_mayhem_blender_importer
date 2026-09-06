#!/usr/bin/env python3
"""Decompress Nippon Ichi YKCMP_V1 streams.

The game files in this repository may wrap a YKCMP stream in a small container.
Use --offset (0x0c for the current .nmbm sample) or --scan to locate it.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


MAGIC = b"YKCMP_V1"
HEADER_SIZE = 0x14


def decompress(data: bytes, offset: int = 0) -> bytes:
    if data[offset : offset + 8] != MAGIC:
        raise ValueError(f"YKCMP_V1 magic not found at 0x{offset:x}")
    kind, compressed_size, decompressed_size = struct.unpack_from(
        "<III", data, offset + 8
    )
    if compressed_size < HEADER_SIZE:
        raise ValueError("invalid compressed size")
    end = offset + compressed_size
    if end > len(data):
        raise ValueError(
            f"truncated stream: header needs {compressed_size} bytes, "
            f"only {len(data) - offset} available"
        )

    source = memoryview(data)[offset + HEADER_SIZE : end]
    if kind in (8, 9):
        return _decompress_lz4_block(source, decompressed_size)
    if kind not in (4, 7):
        raise ValueError(f"unsupported YKCMP compression type {kind}")

    output = bytearray()
    cursor = 0

    while cursor < len(source) and len(output) < decompressed_size:
        control = source[cursor]
        cursor += 1
        if control < 0x80:
            literal_end = cursor + control
            if literal_end > len(source):
                raise ValueError("literal extends past compressed stream")
            output.extend(source[cursor:literal_end])
            cursor = literal_end
            continue

        if control < 0xC0:
            length = (control >> 4) - 7
            distance = (control & 0x0F) + 1
        elif control < 0xE0:
            if cursor >= len(source):
                raise ValueError("short two-byte back-reference")
            length = control - 0xC0 + 2
            distance = source[cursor] + 1
            cursor += 1
        else:
            if cursor + 1 >= len(source):
                raise ValueError("short three-byte back-reference")
            second, third = source[cursor], source[cursor + 1]
            cursor += 2
            length = (control << 4) + (second >> 4) - 0xE00 + 3
            distance = ((second & 0x0F) << 8) + third + 1

        if distance > len(output):
            raise ValueError(f"invalid back-reference distance {distance}")
        # Byte-at-a-time copying intentionally supports overlapping matches.
        for _ in range(length):
            output.append(output[-distance])

    if len(output) != decompressed_size:
        raise ValueError(
            f"size mismatch: expected {decompressed_size}, got {len(output)}"
        )
    return bytes(output)


def _decompress_lz4_block(source: memoryview, expected_size: int) -> bytes:
    """Decode the raw LZ4 block used by YKCMP type 8."""
    output = bytearray()
    cursor = 0

    def extended_length(length: int) -> int:
        nonlocal cursor
        if length != 15:
            return length
        while True:
            if cursor >= len(source):
                raise ValueError("truncated LZ4 length")
            value = source[cursor]
            cursor += 1
            length += value
            if value != 255:
                return length

    while cursor < len(source):
        token = source[cursor]
        cursor += 1
        literal_length = extended_length(token >> 4)
        literal_end = cursor + literal_length
        if literal_end > len(source):
            raise ValueError("LZ4 literal extends past input")
        output.extend(source[cursor:literal_end])
        cursor = literal_end
        if cursor == len(source):
            break
        if cursor + 2 > len(source):
            raise ValueError("truncated LZ4 match offset")
        distance = source[cursor] | (source[cursor + 1] << 8)
        cursor += 2
        if distance == 0 or distance > len(output):
            raise ValueError(f"invalid LZ4 match distance {distance}")
        match_length = extended_length(token & 0x0F) + 4
        for _ in range(match_length):
            output.append(output[-distance])

    if len(output) != expected_size:
        raise ValueError(
            f"LZ4 size mismatch: expected {expected_size}, got {len(output)}"
        )
    return bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--offset", type=lambda value: int(value, 0), default=0)
    location.add_argument("--scan", action="store_true")
    args = parser.parse_args()

    source = args.input.read_bytes()
    offset = source.find(MAGIC) if args.scan else args.offset
    if offset < 0:
        raise SystemExit("YKCMP_V1 stream not found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(decompress(source, offset))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

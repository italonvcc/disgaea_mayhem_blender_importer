"""Portable NLTX -> RGBA/PNG, with no Pillow or platform binaries required.

BC1/BC3/BC7 decoding is adapted from Pillow's CC0 BcnDecode.c.
See docs/THIRD_PARTY.md. Only the top mip level is imported.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

if __package__:
    from .bc7_tables import SUBSETS_2, SUBSETS_3, ANCHORS_2, ANCHORS_3A, ANCHORS_3B
    from .ykcmp import decompress
else:
    from bc7_tables import SUBSETS_2, SUBSETS_3, ANCHORS_2, ANCHORS_3A, ANCHORS_3B
    from ykcmp import decompress


# subsets, partition, rotation, selector, color, alpha, endpoint/shared p-bits,
# first and second index precision.
MODES = (
    (3, 4, 0, 0, 4, 0, 1, 0, 3, 0),
    (2, 6, 0, 0, 6, 0, 0, 1, 3, 0),
    (3, 6, 0, 0, 5, 0, 0, 0, 2, 0),
    (2, 6, 0, 0, 7, 0, 1, 0, 2, 0),
    (1, 0, 2, 1, 5, 6, 0, 0, 2, 3),
    (1, 0, 2, 0, 7, 8, 0, 0, 2, 2),
    (1, 0, 0, 0, 7, 7, 1, 0, 4, 0),
    (2, 6, 0, 0, 5, 5, 1, 0, 2, 0),
)
WEIGHTS = {
    2: (0, 21, 43, 64),
    3: (0, 9, 18, 27, 37, 46, 55, 64),
    4: (0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64),
}


def decode_bc7(block):
    if len(block) != 16:
        raise ValueError("BC7 blocks must contain 16 bytes")
    if not block[0]:
        raise ValueError("invalid BC7 block: no mode bit")
    value = int.from_bytes(block, "little")
    mode = (block[0] & -block[0]).bit_length() - 1
    ns, pb, rb, sb, cb, ab, epb, spb, ib, ib2 = MODES[mode]
    bit = mode + 1

    def take(count):
        nonlocal bit
        result = (value >> bit) & ((1 << count) - 1)
        bit += count
        return result

    partition, rotation, selector = take(pb), take(rb), take(sb)
    endpoints = [[0, 0, 0, 255] for _ in range(ns * 2)]
    for channel in range(3):
        for endpoint in endpoints:
            endpoint[channel] = take(cb)
    if ab:
        for endpoint in endpoints:
            endpoint[3] = take(ab)
    if epb or spb:
        step = 1 if epb else 2
        for start in range(0, len(endpoints), step):
            p = take(1)
            for endpoint in endpoints[start:start + step]:
                for channel in range(4 if ab else 3):
                    endpoint[channel] = (endpoint[channel] << 1) | p
        cb += 1
        ab += bool(ab)
    for endpoint in endpoints:
        for channel in range(4 if ab else 3):
            bits = ab if channel == 3 else cb
            v = endpoint[channel] << (8 - bits)
            endpoint[channel] = v | (v >> bits)

    anchors = {0}
    if ns == 2:
        anchors.add(ANCHORS_2[partition])
    elif ns == 3:
        anchors.update((ANCHORS_3A[partition], ANCHORS_3B[partition]))
    first = [take(ib - (i in anchors)) for i in range(16)]
    second = [take(ib2 - (i == 0)) for i in range(16)] if ib2 else first
    cw, aw = WEIGHTS[ib], WEIGHTS[ib2 or ib]
    result = bytearray()
    for i in range(16):
        subset = ((SUBSETS_2[partition] >> i) & 1) if ns == 2 else (
            ((SUBSETS_3[partition] >> (2 * i)) & 3) if ns == 3 else 0)
        c, a = cw[first[i]], aw[second[i]]
        if selector:
            c, a = a, c
        e0, e1 = endpoints[2 * subset:2 * subset + 2]
        color = [((64 - w) * e0[ch] + w * e1[ch] + 32) >> 6
                 for ch, w in enumerate((c, c, c, a))]
        if rotation:
            color[rotation - 1], color[3] = color[3], color[rotation - 1]
        result.extend(color)
    return bytes(result)


def decode_bc1(block, separate_alpha=False):
    c0, c1, indices = struct.unpack("<HHI", block)

    def rgb565(c):
        r, g, b = (c >> 11) & 31, (c >> 5) & 63, c & 31
        return ((r << 3) | (r >> 2), (g << 2) | (g >> 4),
                (b << 3) | (b >> 2), 255)

    colors = [rgb565(c0), rgb565(c1)]
    if c0 > c1 or separate_alpha:
        colors.extend(tuple((a * x + b * y) // 3 for x, y in zip(*colors[:2]))
                      for a, b in ((2, 1), (1, 2)))
    else:
        colors.extend((tuple((x + y) // 2 for x, y in zip(*colors)), (0, 0, 0, 0)))
    return bytes(c for i in range(16) for c in colors[(indices >> (2 * i)) & 3])


def decode_bc3(block):
    result = bytearray(decode_bc1(block[8:], separate_alpha=True))
    a0, a1 = block[:2]
    alpha = [a0, a1]
    if a0 > a1:
        alpha.extend(((7 - i) * a0 + i * a1) // 7 for i in range(1, 7))
    else:
        alpha.extend(((5 - i) * a0 + i * a1) // 5 for i in range(1, 5))
        alpha.extend((0, 255))
    indices = int.from_bytes(block[2:8], "little")
    for i in range(16):
        result[4 * i + 3] = alpha[(indices >> (3 * i)) & 7]
    return bytes(result)


def decode_blocks(pixels, width, height, block_format):
    decoder = {1: decode_bc1, 3: decode_bc3, 7: decode_bc7}[block_format]
    size = 8 if block_format == 1 else 16
    columns, rows = (width + 3) // 4, (height + 3) // 4
    if len(pixels) < columns * rows * size:
        raise ValueError("truncated block-compressed texture")
    result = bytearray(width * height * 4)
    # Atlases have many identical background blocks. Keep a bounded per-image
    # cache; it also avoids retaining asset data across imports.
    cache = {}
    for by in range(rows):
        for bx in range(columns):
            offset = (by * columns + bx) * size
            block = pixels[offset:offset + size]
            decoded = cache.get(block)
            if decoded is None:
                decoded = decoder(block)
                if len(cache) < 65536:
                    cache[block] = decoded
            count = min(4, width - bx * 4) * 4
            for row in range(min(4, height - by * 4)):
                dest = ((by * 4 + row) * width + bx * 4) * 4
                result[dest:dest + count] = decoded[row * 16:row * 16 + count]
    return bytes(result)


def decode_rgba(source: Path):
    data = source.read_bytes()
    if len(data) < 0x38 or data[:8] != b"NMPLTEX1":
        raise ValueError("not a complete NMPLTEX1 texture")
    texture_type = data[0x14]
    bits = struct.unpack_from("<H", data, 0x16)[0]
    width, height = struct.unpack_from("<II", data, 0x18)
    if not width or not height or width * height > 8192 * 8192:
        raise ValueError("invalid or unsupported texture dimensions (maximum 64M pixels)")
    raw_size, compressed_size, offset = struct.unpack_from("<III", data, 0x2C)
    if offset < 0x38 or compressed_size < 20 or offset + compressed_size > len(data):
        raise ValueError("truncated texture payload")
    if raw_size > width * height * 8 + 1024:
        raise ValueError("texture payload exceeds supported 2D mip-chain size")
    # Slice so the decompressor cannot consume bytes outside this payload.
    pixels = decompress(data[offset:offset + compressed_size])
    if len(pixels) != raw_size:
        raise ValueError("decompressed texture size does not match NLTX header")
    if texture_type == 1 or (texture_type == 2 and bits == 512):
        if len(pixels) < width * height * 4:
            raise ValueError("truncated RGBA texture")
        return width, height, pixels[:width * height * 4]
    if texture_type == 2 and bits == 64:
        block_format = 1
    elif texture_type == 3 or (texture_type == 2 and bits == 128):
        block_format = 3
    elif texture_type == 6:
        block_format = 7
    else:
        raise ValueError(f"unsupported texture type={texture_type:#x}, bits={bits}")
    return width, height, decode_blocks(pixels, width, height, block_format)


def png_bytes(width, height, rgba):
    """Encode exact 8-bit RGBA using standard-library zlib."""
    if len(rgba) != width * height * 4:
        raise ValueError("RGBA byte count does not match image dimensions")

    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    stride = width * 4
    scanlines = b"".join(b"\0" + rgba[y * stride:(y + 1) * stride] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines))
            + chunk(b"IEND", b""))

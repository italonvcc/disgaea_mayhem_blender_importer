"""Read named, sparse blend-shape position deltas from the NMB MESH tables."""

from dataclasses import dataclass
import math
import struct

if __package__:
    from .nmbm_to_obj import chunks, parse_mesh
else:
    from nmbm_to_obj import chunks, parse_mesh


@dataclass(frozen=True)
class Morph:
    name: str
    default: float
    # Vertex indices address the concatenation of all submeshes.
    deltas: tuple[tuple[int, tuple[float, float, float]], ...]


def read_morphs(data):
    found = [c for c in chunks(data) if c[0] == b"MESH"]
    if len(found) != 1:
        raise ValueError("expected one MESH chunk for morphs")
    _, offset, header, size = found[0]
    meshes, vertex_base, index_base = parse_mesh(data, offset, header)
    base, end = offset + header, offset + size
    node_count, target_count, index_size = struct.unpack_from("<3I", data, base + 0x20)
    if node_count == target_count == 0:
        return ()
    names = base + 0x40 + len(meshes) * 0x50
    targets = names + node_count * 12
    records = targets + target_count * 12
    strings = base + struct.unpack_from("<I", data, base + 0x38)[0]
    if (node_count < 1 or target_count >= node_count or records > vertex_base
            or not index_base + index_size <= strings < end):
        raise ValueError("invalid MESH morph table bounds")
    nodes, seen_names = [], set()
    for i in range(node_count):
        parent, default, name_offset = struct.unpack_from("<ifI", data, names + i * 12)
        start = strings + name_offset
        stop = data.find(b"\0", start, end) if start < end else -1
        if stop <= start or not math.isfinite(default) or not 0 <= default <= 1:
            raise ValueError("invalid MESH morph name or default")
        name = data[start:stop].decode("utf-8")
        # Observed hierarchy: a unit BlendShape envelope and named children.
        if (i == 0 and (parent != -1 or default != 1 or name != "BlendShape")
                or i > 0 and parent != 0 or name in seen_names):
            raise ValueError("unsupported MESH morph hierarchy")
        nodes.append((name, default))
        seen_names.add(name)
    total_vertices = sum(m.vertex_count for m in meshes)
    deltas, cursor = {}, 0
    for i in range(target_count):
        identifier, start, count = struct.unpack_from("<3I", data, targets + i * 12)
        stop = start + count
        if (not 0 < identifier < node_count or identifier in deltas or start != cursor
                or records + stop * 40 > vertex_base):
            raise ValueError("invalid MESH morph delta range")
        values, seen_vertices = [], set()
        for j in range(start, stop):
            vertex, *vectors = struct.unpack_from("<I9f", data, records + j * 40)
            if (vertex >= total_vertices or vertex in seen_vertices
                    or not all(math.isfinite(v) for v in vectors)):
                raise ValueError("invalid MESH morph vertex or delta")
            seen_vertices.add(vertex)
            values.append((vertex, tuple(vectors[:3])))
        deltas[identifier] = tuple(values)
        cursor = stop
    if vertex_base - (records + cursor * 40) >= 0x30:
        raise ValueError("unexpected data after MESH morph deltas")
    # Some fox controls have no delta record. Preserve them as empty controls
    # so the shared animation set has the same names on both variants.
    return tuple(Morph(name, default, deltas.get(i, ()))
                 for i, (name, default) in enumerate(nodes) if i)

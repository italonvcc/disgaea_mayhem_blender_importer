"""Read NODE bones and map mesh influences to the NMB skin palette."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

if __package__:
    from .nmbm_to_obj import VertexData, chunks
else:
    from nmbm_to_obj import VertexData, chunks


@dataclass
class Node:
    identifier: int
    name: str
    parent: int
    skin_index: int
    scale: tuple
    pre_rotation: tuple
    translation: tuple
    rotation: tuple
    inverse_bind: tuple | None


def read_nodes(data: bytes) -> dict[int, Node]:
    node_chunks = [chunk for chunk in chunks(data) if chunk[0] == b"NODE"]
    if len(node_chunks) != 1:
        raise ValueError("expected one NODE chunk for a rigged model")
    _, start, header_size, size = node_chunks[0]
    payload, end = start + header_size, start + size
    count, skin_count = struct.unpack_from("<HH", data, payload + 0x14)
    cursor = payload + 0x20
    nodes = {}
    for _ in range(count):
        identifier, _flags, name_offset, skin_index, parent = struct.unpack_from(
            "<HHIii", data, cursor
        )
        record_size = 0xE0 if skin_index >= 0 else 0xA0
        if cursor + record_size > end or skin_index < -1:
            raise ValueError("invalid NODE record")
        name_start = cursor + name_offset
        if not payload <= name_start < end:
            raise ValueError("NODE name offset is out of range")
        name_end = data.find(b"\0", name_start, end)
        if name_end < 0:
            raise ValueError("unterminated NODE name")
        name = data[name_start:name_end].decode("utf-8")
        if identifier in nodes:
            raise ValueError(f"duplicate node ID {identifier}")
        nodes[identifier] = Node(
            identifier, name, parent, skin_index,
            struct.unpack_from("<3f", data, cursor + 0x20),
            struct.unpack_from("<4f", data, cursor + 0x2C),
            struct.unpack_from("<3f", data, cursor + 0x3C),
            struct.unpack_from("<4f", data, cursor + 0x48),
            struct.unpack_from("<16f", data, cursor + 0xA0)
            if skin_index >= 0 else None,
        )
        cursor += record_size

    palette = [node.skin_index for node in nodes.values() if node.skin_index >= 0]
    if sorted(palette) != list(range(skin_count)) or not palette:
        raise ValueError("missing or duplicate NODE skin palette entries")
    for node in nodes.values():
        seen = {node.identifier}
        parent = node.parent
        while parent != -1:
            if parent not in nodes or parent in seen:
                raise ValueError(f"invalid parent hierarchy for {node.name}")
            seen.add(parent)
            parent = nodes[parent].parent
    return nodes


def skeleton_nodes(nodes: dict[int, Node]) -> list[Node]:
    """Include deform bones, their ancestors, and their endpoint/attachment nodes."""
    wanted = {i for i, node in nodes.items() if node.skin_index >= 0}
    for identifier, node in nodes.items():
        parent = node.parent
        while parent != -1:
            if nodes[parent].skin_index >= 0:
                wanted.add(identifier)
                break
            parent = nodes[parent].parent
    for identifier in list(wanted):
        parent = nodes[identifier].parent
        while parent != -1:
            wanted.add(parent)
            parent = nodes[parent].parent
    return [node for identifier, node in nodes.items() if identifier in wanted]


def vertex_influences(vertices: VertexData, nodes: dict[int, Node]) -> list[dict[str, float]]:
    """Map byte palette indices to bone names, not to NODE IDs or list positions."""
    palette = {node.skin_index: node.name for node in nodes.values() if node.skin_index >= 0}
    if not (len(vertices.positions) == len(vertices.weights) == len(vertices.bone_indices)):
        raise ValueError("mesh is missing four-component skin weights or indices")
    result = []
    for vertex, (weights, indices) in enumerate(zip(vertices.weights, vertices.bone_indices)):
        influences = {}
        for weight, index in zip(weights, indices):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"invalid weight at vertex {vertex}")
            if weight == 0:
                continue  # Unused indices do not have to address the palette.
            if index not in palette:
                raise ValueError(f"unknown skin palette index {index} at vertex {vertex}")
            name = palette[index]
            influences[name] = influences.get(name, 0.0) + weight
        total = sum(influences.values())
        if not 0.98 <= total <= 1.02:
            raise ValueError(f"invalid weight sum {total} at vertex {vertex}")
        result.append({name: weight / total for name, weight in influences.items()})
    return result

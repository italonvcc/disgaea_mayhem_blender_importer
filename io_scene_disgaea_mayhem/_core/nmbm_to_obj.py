#!/usr/bin/env python3
"""Export the static geometry in an NMB/NMBM file to Wavefront OBJ.

This is intentionally a first-stage exporter: positions, normals, UVs,
submeshes, and material slots are preserved. Skinning and animation are not.
Compressed NMBM inputs are decompressed automatically.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .ykcmp import MAGIC as YKCMP_MAGIC
    from .ykcmp import decompress as decompress_ykcmp
else:
    from ykcmp import MAGIC as YKCMP_MAGIC
    from ykcmp import decompress as decompress_ykcmp


@dataclass
class Submesh:
    identifier: int
    cluster_start: int
    vertex_offset: int
    vertex_count: int
    index_offset: int
    index_count: int
    index_width: int
    flags: int
    cluster_count: int = 1


@dataclass
class VertexData:
    positions: list
    normals: list
    uvs: list
    weights: list
    bone_indices: list


def chunks(data: bytes):
    cursor = 0x20
    while cursor + 0x20 <= len(data):
        magic, _version, size, _unknown, header_size = struct.unpack_from(
            "<4s4I", data, cursor
        )
        if size < 0x20 or cursor + size > len(data):
            raise ValueError(f"invalid {magic!r} chunk size at 0x{cursor:x}")
        yield magic, cursor, header_size, size
        cursor += size
        if magic == b"EOFC":
            break


def parse_mesh(data: bytes, chunk_offset: int, header_size: int):
    payload = chunk_offset + header_size
    chunk_size = struct.unpack_from("<I", data, chunk_offset + 8)[0]
    chunk_end = chunk_offset + chunk_size
    if header_size < 0x20 or payload + 0x40 > chunk_end or chunk_end > len(data):
        raise ValueError("invalid MESH chunk bounds")
    submesh_count = struct.unpack_from("<I", data, payload + 0x14)[0]
    _section1_count, _section2_count, index_size, index_relative = (
        struct.unpack_from("<4I", data, payload + 0x20)
    )
    cursor = payload + 0x40
    if cursor + submesh_count * 0x50 > chunk_end:
        raise ValueError("truncated MESH submesh table")
    submeshes: list[Submesh] = []
    for _ in range(submesh_count):
        values = struct.unpack_from("<9I", data, cursor)
        flags = struct.unpack_from("<I", data, cursor + 0x4C)[0]
        submeshes.append(
            Submesh(
                identifier=values[0],
                cluster_start=values[2],
                vertex_offset=values[4],
                vertex_count=values[5],
                index_offset=values[6],
                index_count=values[7],
                index_width=values[8],
                flags=flags,
                cluster_count=values[3],
            )
        )
        cursor += 0x50

    # The header explicitly locates the vertex section relative to the payload.
    # Walking the preceding variable-size tables missed alignment padding.
    vertex_relative = struct.unpack_from("<I", data, payload + 0x3C)[0]
    vertex_base = payload + vertex_relative
    index_base = vertex_base + index_relative
    if vertex_base < cursor or index_base + index_size > chunk_end:
        raise ValueError("vertex/index section lies outside the MESH chunk")
    # Individual index buffers may be separated by alignment padding. Flora
    # Beast has a 68,742-byte buffer followed by two zero bytes. Validate each
    # declared range instead of requiring sum(buffer lengths) == section size.
    previous_end = 0
    for mesh in sorted(submeshes, key=lambda m: m.index_offset):
        if mesh.index_width not in (2, 4):
            raise ValueError(f"unsupported index width {mesh.index_width}")
        end = mesh.index_offset + mesh.index_count * mesh.index_width
        if mesh.index_offset % mesh.index_width:
            raise ValueError("misaligned submesh index offset")
        if mesh.index_offset < previous_end:
            raise ValueError("overlapping submesh index buffers")
        if end > index_size:
            raise ValueError("submesh index buffer exceeds the index section")
        if mesh.index_count % 3:
            raise ValueError("submesh index count is not triangular")
        if not 0 <= mesh.vertex_offset < index_relative or not mesh.vertex_count:
            raise ValueError("invalid submesh vertex range")
        previous_end = end
    return submeshes, vertex_base, index_base


def read_vertex_data(data: bytes, base: int, mesh: Submesh) -> VertexData:
    # Attribute arrays are padded to multiples of 0x30 bytes, relative to
    # the vertex section. This includes colors, UVs, weights and bone indices.
    cursor = mesh.vertex_offset
    count = mesh.vertex_count
    if mesh.flags & ~0x3F:
        raise ValueError(f"unsupported vertex flags {mesh.flags:#x}")

    def vectors(fmt: str, width: int):
        nonlocal cursor
        result = [
            struct.unpack_from(fmt, data, base + cursor + i * width)
            for i in range(count)
        ]
        cursor = ((cursor + count * width + 0x2F) // 0x30) * 0x30
        return result

    positions = vectors("<3f", 12) if mesh.flags & 0x01 else []
    if mesh.flags & 0x08:  # RGBA8 vertex colors
        vectors("<4B", 4)
    normals = vectors("<3f", 12) if mesh.flags & 0x02 else []
    if mesh.flags & 0x04:  # float3 tangent, currently unused
        vectors("<3f", 12)
    uvs = vectors("<2e", 4) if mesh.flags & 0x10 else []
    weights = vectors("<4e", 8) if mesh.flags & 0x20 else []
    bone_indices = vectors("<4B", 4) if mesh.flags & 0x20 else []
    return VertexData(positions, normals, uvs, weights, bone_indices)


def read_vertices(data: bytes, base: int, mesh: Submesh):
    vertices = read_vertex_data(data, base, mesh)
    return vertices.positions, vertices.normals, vertices.uvs


def read_indices(data: bytes, base: int, mesh: Submesh):
    if mesh.index_width not in (2, 4):
        raise ValueError(f"unsupported index width {mesh.index_width}")
    code = "H" if mesh.index_width == 2 else "I"
    return struct.unpack_from(
        f"<{mesh.index_count}{code}", data, base + mesh.index_offset
    )


def load_nmb(path: Path) -> bytes:
    data = path.read_bytes()
    if data[:4] == b"NMB\0":
        return data
    location = data.find(YKCMP_MAGIC)
    if location < 0:
        raise ValueError("input is neither an NMB file nor a YKCMP-wrapped NMBM")
    data = decompress_ykcmp(data, location)
    if data[:4] != b"NMB\0":
        raise ValueError("decompressed data does not start with NMB magic")
    return data


def export_obj(source: Path, destination: Path, texture: Path | None = None) -> None:
    if __package__:
        from .nmbm_materials import read_clusters, mesh_clusters
    else:
        from nmbm_materials import read_clusters, mesh_clusters
    data = load_nmb(source)
    clusters = read_clusters(data)
    mesh_chunks = [item for item in chunks(data) if item[0] == b"MESH"]
    if len(mesh_chunks) != 1:
        raise ValueError(f"expected one MESH chunk, found {len(mesh_chunks)}")
    _, chunk_offset, header_size, _ = mesh_chunks[0]
    submeshes, vertex_base, index_base = parse_mesh(data, chunk_offset, header_size)

    lines = [f"# Static export from {source.name}", "# Generated by tools/nmbm_to_obj.py"]
    if texture is not None:
        lines.append(f"mtllib {destination.with_suffix('.mtl').name}")
    vertex_start = 1
    total_triangles = 0
    for mesh in submeshes:
        positions, normals, uvs = read_vertices(data, vertex_base, mesh)
        indices = read_indices(data, index_base, mesh)
        if len(indices) % 3:
            raise ValueError(f"submesh {mesh.identifier} index count is not triangular")
        if indices and max(indices) >= mesh.vertex_count:
            raise ValueError(f"submesh {mesh.identifier} has an out-of-range index")

        lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in positions)
        lines.extend(f"vt {u:.9g} {1.0 - v:.9g}" for u, v in uvs)
        lines.extend(f"vn {x:.9g} {y:.9g} {z:.9g}" for x, y, z in normals)
        lines.append(f"o submesh_{mesh.identifier}")
        material_starts = {c.index_start: c.material for c in mesh_clusters(mesh, clusters)}

        for face in range(0, len(indices), 3):
            if face in material_starts:
                lines.append(f"usemtl material_{material_starts[face]}")
            corners = []
            for local_index in indices[face : face + 3]:
                index = vertex_start + local_index
                if uvs and normals:
                    corners.append(f"{index}/{index}/{index}")
                elif normals:
                    corners.append(f"{index}//{index}")
                elif uvs:
                    corners.append(f"{index}/{index}")
                else:
                    corners.append(str(index))
            lines.append("f " + " ".join(corners))
        vertex_start += mesh.vertex_count
        total_triangles += len(indices) // 3

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if texture is not None:
        material_ids = sorted({c.material for mesh in submeshes for c in mesh_clusters(mesh, clusters)})
        relative_texture = texture.name
        mtl_lines = ["# Material placeholders for the selected diffuse atlas"]
        for material_id in material_ids:
            mtl_lines.extend(
                [
                    f"newmtl material_{material_id}",
                    "Kd 1 1 1",
                    f"map_Kd {relative_texture}",
                    "",
                ]
            )
        destination.with_suffix(".mtl").write_text(
            "\n".join(mtl_lines), encoding="utf-8"
        )
    print(
        f"wrote {destination}: {vertex_start - 1} vertices, "
        f"{total_triangles} triangles, {len(submeshes)} submeshes"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--texture",
        type=Path,
        help="optional diffuse PNG placed alongside the OBJ",
    )
    args = parser.parse_args()
    export_obj(args.input, args.output, args.texture)


if __name__ == "__main__":
    main()

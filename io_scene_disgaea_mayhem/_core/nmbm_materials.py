"""Read CLUS triangle ranges and MATE albedo references."""

from dataclasses import dataclass
from pathlib import Path
import struct

if __package__:
    from .nmbm_to_obj import chunks
else:
    from nmbm_to_obj import chunks


@dataclass(frozen=True)
class Cluster:
    identifier: int
    material: int
    index_start: int
    index_count: int


@dataclass(frozen=True)
class Material:
    identifier: int
    name: str
    technique: str
    albedo: str | None


def section(data, magic):
    found = [c for c in chunks(data) if c[0] == magic]
    if len(found) != 1:
        raise ValueError(f"expected one {magic.decode()} chunk")
    _, offset, header, size = found[0]
    if not 0x20 <= header <= size:
        raise ValueError("invalid material/cluster chunk header")
    return offset + header, offset + size


def bounded_unpack(data, fmt, offset, end):
    if offset < 0 or offset + struct.calcsize(fmt) > end:
        raise ValueError("truncated material/cluster data")
    return struct.unpack_from(fmt, data, offset)


def read_clusters(data):
    base, end = section(data, b"CLUS")
    offset, stride, _, count = bounded_unpack(data, "<4I", base + 0x10, end)
    if offset < 0x30 or stride != 20 or base + offset + stride * count > end:
        raise ValueError("invalid CLUS record table")
    result = []
    for i in range(count):
        identifier, _, material, length, start = bounded_unpack(data, "<5I", base + offset + i * stride, end)
        if identifier != i or start % 3 or length % 3:
            raise ValueError("invalid CLUS triangle range")
        result.append(Cluster(identifier, material, start, length))
    return result


def mesh_clusters(mesh, clusters):
    start, stop = mesh.cluster_start, mesh.cluster_start + mesh.cluster_count
    if mesh.cluster_count < 1 or stop > len(clusters):
        raise ValueError("submesh CLUS range is out of bounds")
    result = clusters[start:stop]
    cursor = 0
    for cluster in sorted(result, key=lambda c: c.index_start):
        if cluster.index_start != cursor:
            raise ValueError("CLUS triangle ranges overlap or leave gaps")
        cursor += cluster.index_count
    if cursor != mesh.index_count:
        raise ValueError("CLUS triangle ranges do not cover the submesh")
    return result


def read_materials(data):
    base, end = section(data, b"MATE")
    stride, count, _, offset = bounded_unpack(data, "<4I", base + 0x10, end)
    if stride != 0x38 or offset < 0x24 or base + offset + stride * count > end:
        raise ValueError("unsupported MATE record layout")

    def string(relative):
        start = base + relative
        if not base <= start < end:
            raise ValueError("MATE string offset is out of bounds")
        stop = data.find(b"\0", start, end)
        if stop < 0:
            raise ValueError("unterminated MATE string")
        return data[start:stop].decode("utf-8")

    result = {}
    for i in range(count):
        record = bounded_unpack(data, "<14I", base + offset + i * stride, end)
        identifier, name = record[0], string(record[2])
        properties, size = base + record[7], record[8]
        if properties < base or properties + size > end:
            raise ValueError("MATE property block is out of bounds")
        limit = properties + size
        _, _, technique, property_count = bounded_unpack(data, "<4I", properties, limit)
        technique = string(technique)
        cursor, albedo = properties + 16, None
        for _ in range(property_count):
            kind, prop_name = bounded_unpack(data, "<II", cursor, limit)
            prop_name = string(prop_name)
            kind &= 0xFFFF  # High half identifies a texture sampler slot.
            if kind == 10:
                reference = bounded_unpack(data, "<I", cursor + 8, limit)[0]
                texture = string(reference)
                if prop_name == "AlbedoSampler":
                    albedo = texture
                cursor += 12
            elif kind in (5, 6, 7, 8):
                cursor += (kind - 2) * 4  # two header words + 1..4 floats
            else:
                raise ValueError(f"unsupported MATE property type {kind}: {prop_name}")
            if cursor > limit:
                raise ValueError("truncated MATE property value")
        if cursor != limit or identifier in result:
            raise ValueError("invalid MATE property block or duplicate material ID")
        result[identifier] = Material(identifier, name, technique, albedo)
    return result


def find_albedo_files(source: Path, materials):
    """Match case-insensitive basenames; never follow embedded relative paths."""
    files = {p.name.casefold(): p for p in source.parent.iterdir() if p.is_file()}
    resolved, missing = {}, []
    for identifier, material in materials.items():
        if material.albedo is None:
            continue
        name = material.albedo.replace("\\", "/").rsplit("/", 1)[-1]
        stem = Path(name).stem
        path = next((files[n.casefold()] for n in (stem + ".png", name)
                     if n.casefold() in files), None)
        if path:
            resolved[identifier] = path
        else:
            missing.append(name)
    return resolved, sorted(set(missing))

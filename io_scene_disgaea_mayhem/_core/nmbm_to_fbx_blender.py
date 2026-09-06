"""Import NMBM geometry and skin into Blender and export a rigged FBX.

blender --background --python-exit-code 1 --python tools/nmbm_to_fbx_blender.py -- \
    input.nmbm output.fbx --texture output/diffuse.png --blend output/rig.blend

Use --animation clip.nmba or --animation-dir directory to include motion.
Run this with Blender's Python; it needs no third-party Python packages.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

if __package__:
    from .nmbm_morphs import read_morphs
    from .morphs_to_blender import add_shape_keys, sync_facial_export
    from .nmbm_to_obj import chunks, load_nmb, parse_mesh, read_indices, read_vertex_data
    from .nmbm_rig import read_nodes, skeleton_nodes, vertex_influences
    from .nmbm_materials import read_clusters, mesh_clusters, read_materials
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nmbm_morphs import read_morphs
    from morphs_to_blender import add_shape_keys, sync_facial_export
    from nmbm_to_obj import chunks, load_nmb, parse_mesh, read_indices, read_vertex_data
    from nmbm_rig import read_nodes, skeleton_nodes, vertex_influences
    from nmbm_materials import read_clusters, mesh_clusters, read_materials


# Game Y-up -> Blender Z-up, with the character facing Blender -Y.
GAME_TO_BLENDER = Matrix.Rotation(math.pi / 2, 4, "X")


def bind_matrices(nodes):
    """Use stored skin binds; the authored default pose can differ from them."""
    worlds = {}

    def world(identifier):
        if identifier in worlds:
            return worlds[identifier]
        node = nodes[identifier]
        if any(abs(a - 1) > 1e-5 for a in node.scale):
            raise ValueError(f"non-unit NODE scale is not supported: {node.name}")
        x, y, z, w = node.rotation
        px, py, pz, pw = node.pre_rotation
        if not all(math.isfinite(v) for v in (*node.translation, x, y, z, w, px, py, pz, pw)):
            raise ValueError(f"non-finite NODE transform: {node.name}")
        rotation = Quaternion((w, x, y, z))
        pose = Quaternion((pw, px, py, pz))
        if abs(rotation.magnitude - 1) > 1e-4 or abs(pose.magnitude - 1) > 1e-4:
            raise ValueError(f"invalid NODE quaternion: {node.name}")
        # NODE +0x48 is joint orientation; +0x2c is the default pose rotation
        # after that orientation (historically named pre_rotation in Node).
        local = (Matrix.Translation(node.translation)
                 @ rotation.normalized().to_matrix().to_4x4()
                 @ pose.normalized().to_matrix().to_4x4())
        worlds[identifier] = world(node.parent) @ local if node.parent != -1 else local
        return worlds[identifier]

    max_error = 0.0
    # First compose the entire authored pose without mixing in skin binds.
    for identifier in nodes:
        world(identifier)
    bindings = dict(worlds)
    for node in nodes.values():
        transform = world(node.identifier)
        if node.inverse_bind is not None:
            stored = Matrix([node.inverse_bind[i:i + 4] for i in range(0, 16, 4)]).transposed()
            if not all(math.isfinite(v) for row in stored for v in row):
                raise ValueError(f"non-finite inverse bind: {node.name}")
            if max(abs(stored[3][c] - (c == 3)) for c in range(4)) > 1e-5:
                raise ValueError(f"non-affine inverse bind: {node.name}")
            rotation = stored.to_3x3()
            orthogonal = rotation.transposed() @ rotation
            if (abs(rotation.determinant() - 1) > 2e-4
                    or max(abs(orthogonal[r][c] - (r == c)) for r in range(3) for c in range(3)) > 2e-4):
                raise ValueError(f"invalid rigid inverse bind: {node.name}")
            product = transform @ stored
            error = max(abs(product[r][c] - (r == c)) for r in range(4) for c in range(4))
            max_error = max(max_error, error)
            # The first fox's TopFrontSideHair1_L is bound at a slightly
            # different orientation from its NODE default. Do not discard
            # that inverse bind or propagate its correction into other bones.
            if error > 2e-4:
                bindings[node.identifier] = stored.inverted()
    return bindings, max_error


def make_armature(name, collection, nodes, worlds):
    skeleton = skeleton_nodes(nodes)
    names = [node.name for node in skeleton]
    if len(set(names)) != len(names):
        raise ValueError("duplicate skeleton names are not supported")
    armature_data = bpy.data.armatures.new(name + "_armature")
    armature = bpy.data.objects.new(name + "_rig", armature_data)
    collection.objects.link(armature)
    armature.show_in_front = True
    armature_data.display_type = "OCTAHEDRAL"
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for node in skeleton:
            bone = armature_data.edit_bones.new(node.name)
            children = [child for child in skeleton if child.parent == node.identifier]
            # NMB joints run along local X; Blender bones run along local Y.
            # Some left-side limbs use negative-X lengths.
            child = next((c for c in children if abs(c.translation[0]) > 1e-5), None)
            length = abs(child.translation[0]) if child else 0.035
            sign = -1 if child and child.translation[0] < 0 else 1
            axes = Matrix.Rotation(-sign * math.pi / 2, 4, "Z")
            bone.head = (0, 0, 0)
            bone.tail = (0, max(length, 0.005), 0)
            bone.matrix = GAME_TO_BLENDER @ worlds[node.identifier] @ axes
            bone.use_deform = node.skin_index >= 0
            bone["nmb_node_id"] = node.identifier
            bone["nmb_skin_index"] = node.skin_index
        for node in skeleton:
            if node.parent != -1:
                bone = armature_data.edit_bones[node.name]
                bone.parent = armature_data.edit_bones[nodes[node.parent].name]
                bone.use_connect = False
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def make_material(identifier, texture, definition=None):
    material = bpy.data.materials.new(definition.name if definition else f"material_{identifier}")
    material.use_nodes = True
    if texture is not None:
        shader = material.node_tree.nodes.get("Principled BSDF")
        image_node = material.node_tree.nodes.new("ShaderNodeTexImage")
        image_node.image = bpy.data.images.load(str(texture.resolve()), check_existing=True)
        material.node_tree.links.new(image_node.outputs["Color"], shader.inputs["Base Color"])
        if definition and definition.technique == "TECHNIQUE_BLEND":
            material.node_tree.links.new(image_node.outputs["Alpha"], shader.inputs["Alpha"])
            material.surface_render_method = "DITHERED"
    return material


def build_scene(source: Path, texture: Path | None = None, *, material_textures=None):
    """Add the model to a new collection; do not delete an existing scene."""
    if texture is not None and not texture.is_file():
        raise ValueError(f"texture not found: {texture}")
    data = load_nmb(source)
    nodes = read_nodes(data)
    clusters = read_clusters(data)
    definitions = read_materials(data)
    worlds, bind_error = bind_matrices(nodes)
    mesh_chunks = [chunk for chunk in chunks(data) if chunk[0] == b"MESH"]
    if len(mesh_chunks) != 1:
        raise ValueError("expected one MESH chunk")
    _, offset, header_size, _ = mesh_chunks[0]
    submeshes, vertex_base, index_base = parse_mesh(data, offset, header_size)
    morphs = read_morphs(data)
    parsed = []
    for submesh in submeshes:
        vertices = read_vertex_data(data, vertex_base, submesh)
        indices = read_indices(data, index_base, submesh)
        if len(indices) % 3 or not indices or max(indices) >= len(vertices.positions):
            raise ValueError(f"invalid triangles in submesh {submesh.identifier}")
        influences = vertex_influences(vertices, nodes)
        groups = mesh_clusters(submesh, clusters)
        if any(group.material not in definitions for group in groups):
            raise ValueError("CLUS references an absent material")
        parsed.append((submesh, vertices, indices, influences, groups))

    collection = bpy.data.collections.new(source.stem)
    bpy.context.scene.collection.children.link(collection)
    armature = make_armature(source.stem, collection, nodes, worlds)
    materials, objects = {}, []
    rotation = GAME_TO_BLENDER.to_3x3()
    vertex_start = 0
    for submesh, vertices, indices, influences, groups in parsed:
        mesh = bpy.data.meshes.new(f"submesh_{submesh.identifier}")
        mesh.from_pydata(
            [rotation @ Vector(position) for position in vertices.positions], [],
            [indices[i:i + 3] for i in range(0, len(indices), 3)],
        )
        mesh.update()
        obj = bpy.data.objects.new(mesh.name, mesh)
        collection.objects.link(obj)
        add_shape_keys(obj, submesh, vertex_start, morphs, rotation)
        vertex_start += submesh.vertex_count
        if vertices.uvs:
            uv_layer = mesh.uv_layers.new(name="UVMap")
            for loop in mesh.loops:
                u, v = vertices.uvs[loop.vertex_index]
                uv_layer.data[loop.index].uv = (u, 1 - v)
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        if vertices.normals:
            mesh.normals_split_custom_set_from_vertices(
                [rotation @ Vector(normal) for normal in vertices.normals]
            )
        slots = {}
        for group in groups:
            identifier = group.material
            if identifier not in materials:
                image = texture if texture else (material_textures or {}).get(identifier)
                materials[identifier] = make_material(identifier, image, definitions[identifier])
            if identifier not in slots:
                slots[identifier] = len(mesh.materials)
                mesh.materials.append(materials[identifier])
            for polygon in mesh.polygons[group.index_start // 3:(group.index_start + group.index_count) // 3]:
                polygon.material_index = slots[identifier]
        groups = {node.name: obj.vertex_groups.new(name=node.name)
                  for node in nodes.values() if node.skin_index >= 0}
        for index, weights in enumerate(influences):
            for name, weight in weights.items():
                groups[name].add([index], weight, "REPLACE")
        obj.parent = armature
        modifier = obj.modifiers.new("NMB skin", "ARMATURE")
        modifier.object = armature
        modifier.use_deform_preserve_volume = False
        objects.append(obj)
    bpy.context.view_layer.update()
    print(f"NMB: {len(armature.data.bones)} bones, "
          f"{sum(node.skin_index >= 0 for node in nodes.values())} deforming; "
          f"default-pose vs skin-bind difference {bind_error:.3g}")
    return armature, objects


def export_fbx(destination: Path, armature, meshes, *, animated=False, use_nla=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in [armature, *meshes]:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    original_name = armature.name
    if animated:
        # Blender prefixes take names with the object name twice on reimport.
        # Keep that label short so its 63-byte ID limit does not cut clip names.
        armature.name = "Rig"
    try:
        with sync_facial_export(armature, animated=animated, use_nla=use_nla):
            _write_fbx(destination, animated, use_nla)
    finally:
        armature.name = original_name


def _write_fbx(destination, animated, use_nla):
    bpy.ops.export_scene.fbx(
        filepath=str(destination.resolve()), use_selection=True,
        object_types={"ARMATURE", "MESH"}, use_mesh_modifiers=False,
        add_leaf_bones=False, use_armature_deform_only=False,
        # FBX has no Blender use_connect flag. On reimport Blender joins
        # Y-aligned chains automatically and then suppresses their animated
        # translations. X-axis FBX joints preserve those translations; the
        # .blend keeps the original, disconnected Y-axis editing rig.
        primary_bone_axis="X" if animated else "Y",
        secondary_bone_axis="Y" if animated else "X",
        bake_anim=animated, bake_anim_use_all_actions=animated and not use_nla,
        bake_anim_use_nla_strips=animated and use_nla, bake_anim_use_all_bones=True,
        bake_anim_step=1.0, bake_anim_simplify_factor=0.0,
        path_mode="COPY", embed_textures=True,
        axis_forward="-Z", axis_up="Y",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--texture", type=Path, help="decoded diffuse PNG, optional")
    parser.add_argument("--blend", type=Path, help="also save an editable .blend")
    parser.add_argument("--animation", type=Path, action="append", default=[],
                        help="NMA/NMBA clip to include; can be repeated")
    parser.add_argument("--animation-dir", type=Path,
                        help="include every .nmba/.nma in this directory")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if args.output.suffix.lower() != ".fbx":
        parser.error("output must use the .fbx extension")
    if args.blend and args.blend.suffix.lower() != ".blend":
        parser.error("--blend must use the .blend extension")
    animation_paths = list(args.animation)
    if args.animation_dir:
        if not args.animation_dir.is_dir():
            parser.error("--animation-dir must be a directory")
        found = sorted(path for path in args.animation_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in (".nmba", ".nma"))
        if not found:
            parser.error("--animation-dir contains no NMA/NMBA clips")
        animation_paths.extend(found)
    animation_paths = list(dict.fromkeys(path.resolve() for path in animation_paths))
    # CLI conversion starts with an empty scene. build_scene itself is additive.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    armature, meshes = build_scene(args.input, args.texture)
    if animation_paths:
        from nmba_to_blender import load_actions
        nodes = read_nodes(load_nmb(args.input))
        worlds, _ = bind_matrices(nodes)
        load_actions(armature, nodes, worlds, animation_paths, GAME_TO_BLENDER)
    export_fbx(args.output, armature, meshes, animated=bool(animation_paths))
    if args.blend:
        args.blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()))
    print(f"Wrote rigged FBX: {args.output}")


if __name__ == "__main__":
    main()

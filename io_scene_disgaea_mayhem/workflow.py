"""Additive import and character-scoped FBX export for Blender."""

from contextlib import contextmanager
from pathlib import Path
import tempfile
import uuid

import bpy

from ._core.nltx_texture import decode_rgba, png_bytes
from ._core.nmba_to_blender import activate_action, load_actions
from ._core.morphs_to_blender import preserve_facial_state
from ._core.nmbm_rig import read_nodes
from ._core.nmbm_materials import read_materials, find_albedo_files
from ._core.nmbm_to_obj import load_nmb
from ._core.nmbm_to_fbx_blender import GAME_TO_BLENDER, bind_matrices, build_scene, export_fbx


def selected_rig(context):
    obj = context.active_object
    if obj and obj.type == "MESH":
        obj = obj.find_armature()
    return obj if obj and obj.type == "ARMATURE" and obj.get("dm_source") else None


def animation_files(directory):
    if not directory.is_dir():
        raise ValueError(f"Animation folder does not exist: {directory}")
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in {".nmba", ".nma"})


def matching_animation_dir(source):
    for directory in (source.parent / "animation", source.parent.parent / "animation"):
        if directory.is_dir():
            return directory
    return None


@contextmanager
def scene_transaction(context):
    """Rollback only data created by a failed import, preserving user work."""
    groups = ("objects", "collections", "meshes", "armatures", "materials", "images", "actions")
    before = {name: set(getattr(bpy.data, name)) for name in groups}
    selected, active = list(context.selected_objects), context.view_layer.objects.active
    scene = context.scene
    timing = (scene.render.fps, scene.render.fps_base, scene.frame_start,
              scene.frame_end, scene.frame_current, scene.frame_subframe)
    try:
        yield
    except Exception:
        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for name in groups:
            group = getattr(bpy.data, name)
            for item in set(group) - before[name]:
                group.remove(item, do_unlink=True)
        for obj in context.selected_objects:
            obj.select_set(False)
        for obj in selected:
            obj.select_set(True)
        context.view_layer.objects.active = active
        scene.render.fps, scene.render.fps_base = timing[:2]
        scene.frame_start, scene.frame_end = timing[2:4]
        scene.frame_set(timing[4], subframe=timing[5])
        raise


def add_clips(context, rig, paths):
    existing = {entry.action.get("dm_clip_source") for entry in rig.dm_clips if entry.action}
    paths = list(dict.fromkeys(Path(p).resolve() for p in paths))
    paths = [p for p in paths if str(p) not in existing]
    if not paths:
        return [], []
    source = Path(bpy.path.abspath(rig["dm_source"]))
    if not source.is_file():
        raise ValueError("Original model file is missing; restore it before adding clips")
    nodes = read_nodes(load_nmb(source))
    worlds, _ = bind_matrices(nodes)
    count = len(rig.dm_clips)
    old_index = rig.dm_clip_index
    animation = rig.animation_data
    old_action = animation.action if animation else None
    old_slot = animation.action_slot if animation else None
    with preserve_facial_state(rig, rollback_only=True), scene_transaction(context):
        try:
            actions = load_actions(rig, nodes, worlds, paths, GAME_TO_BLENDER)
            prior_fps = {float(item.action["nma_fps"]) for item in rig.dm_clips if item.action}
            if prior_fps and any(float(action["nma_fps"]) not in prior_fps for action in actions):
                raise ValueError("Clips on one character must have the same frame rate")
            for path, action in zip(paths, actions):
                action["dm_owner"] = rig["dm_id"]
                action["dm_clip_source"] = str(path)
                entry = rig.dm_clips.add()
                entry.name = path.stem
                entry.action = action
            rig.dm_clip_index = count
            activate_action(rig, actions[0])
        except Exception:
            while len(rig.dm_clips) > count:
                rig.dm_clips.remove(len(rig.dm_clips) - 1)
            rig.dm_clip_index = old_index
            if animation is not None:
                animation.action = old_action
                if old_action:
                    animation.action_slot = old_slot
            else:
                rig.animation_data_clear()
            raise
    extras = sorted({a.get("nma_extra_chunks", "") for a in actions} - {""})
    warnings = ["Extra tracks skipped: " + ", ".join(extras)] if extras else []
    skipped = {path for action in actions for path in action.get("nma_skipped_endpoints", "").splitlines()}
    if skipped:
        warnings.append(f"Skipped {len(skipped)} unused constant endpoints absent from this variant")
    return actions, warnings


def import_character(context, source, *, texture=None, auto_texture=True, paths=()):
    warnings = []
    material_paths = {}
    if texture is None and auto_texture:
        material_paths, missing = find_albedo_files(source, read_materials(load_nmb(source)))
        if missing:
            warnings.append("Missing diffuse textures: " + ", ".join(missing))
    with scene_transaction(context), tempfile.TemporaryDirectory(prefix="disgaea_texture_") as tmp:
        converted = {}

        def prepare(path):
            if path is None or path.suffix.lower() != ".nltx":
                return path
            if path not in converted:
                decoded = Path(tmp) / (path.stem + ".png")
                decoded.write_bytes(png_bytes(*decode_rgba(path)))
                converted[path] = decoded
            return converted[path]

        decoded = prepare(texture)
        material_textures = {identifier: prepare(path) for identifier, path in material_paths.items()}
        rig, meshes = build_scene(source, decoded, material_textures=material_textures)
        rig["dm_source"] = str(source.resolve())
        rig["dm_id"] = uuid.uuid4().hex
        for mesh in meshes:
            for material in mesh.data.materials:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE" and node.image:
                        node.image.pack()
                        if Path(node.image.filepath) in converted.values():
                            # Packed PNG bytes survive temporary-directory cleanup and
                            # are used by Blender's FBX embedded-texture exporter.
                            node.image.filepath = "//" + Path(node.image.filepath).name
        actions, messages = add_clips(context, rig, paths)
        warnings.extend(messages)
        for obj in context.selected_objects:
            obj.select_set(False)
        rig.select_set(True)
        context.view_layer.objects.active = rig
    return rig, meshes, actions, warnings


def export_character(context, destination, rig, clip_mode):
    meshes = [obj for obj in context.scene.objects
              if obj.type == "MESH" and obj.find_armature() == rig]
    if not meshes:
        raise ValueError("No meshes attached to this character in the current scene")
    actions = [entry.action for entry in rig.dm_clips if entry.action]
    if clip_mode == "CURRENT":
        action = rig.animation_data.action if rig.animation_data else None
        actions = [action] if action else []
    elif clip_mode == "NONE":
        actions = []
    # This uses only the chosen rig's clips. The CLI's 'all compatible actions'
    # option is unsafe in a live scene containing more than one character.
    selected, active = list(context.selected_objects), context.view_layer.objects.active
    scene = context.scene
    timing = (scene.frame_start, scene.frame_end, scene.frame_current, scene.frame_subframe,
              scene.render.fps, scene.render.fps_base)
    mode = context.object.mode if context.object else "OBJECT"
    animation = rig.animation_data
    had_animation = animation is not None
    if animation is None:
        animation = rig.animation_data_create()
    old_action, old_slot = animation.action, animation.action_slot
    if animation.use_tweak_mode:
        raise ValueError("Exit NLA Tweak Mode before exporting")
    old_use_nla = animation.use_nla
    old_pose_position = rig.data.pose_position
    old_tracks = [(track, track.mute, track.is_solo) for track in animation.nla_tracks]
    created = []
    pose = {bone.name: bone.matrix_basis.copy() for bone in rig.pose.bones}
    try:
        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        animation.action = None
        animation.use_nla = True
        rig.data.pose_position = "POSE"
        for track, _, _ in old_tracks:
            track.mute = True
            track.is_solo = False
        if actions:
            fps = {float(action.get("nma_fps", scene.render.fps / scene.render.fps_base))
                   for action in actions}
            if len(fps) != 1:
                raise ValueError("Export clips with different frame rates separately")
            fps = fps.pop()
            scene.render.fps = round(fps)
            scene.render.fps_base = round(fps) / fps
            for action in actions:
                track = animation.nla_tracks.new()
                created.append(track)
                track.name = action.name
                strip = track.strips.new(action.name, int(action.frame_start), action)
                strip.action_slot = action.slots[0]
                strip.action_frame_start = action.frame_start
                strip.action_frame_end = action.frame_end
        else:
            # A static export represents the bind pose, even if a clip was active.
            for bone in rig.pose.bones:
                bone.matrix_basis.identity()
        context.view_layer.update()
        export_fbx(destination, rig, meshes, animated=bool(actions), use_nla=bool(actions))
    finally:
        for track in created:
            animation.nla_tracks.remove(track)
        for track, muted, solo in old_tracks:
            track.mute, track.is_solo = muted, solo
        animation.action = old_action
        animation.use_nla = old_use_nla
        rig.data.pose_position = old_pose_position
        if old_action:
            animation.action_slot = old_slot
        for name, matrix in pose.items():
            rig.pose.bones[name].matrix_basis = matrix
        if not had_animation:
            rig.animation_data_clear()
        scene.frame_start, scene.frame_end = timing[:2]
        scene.render.fps, scene.render.fps_base = timing[4:]
        scene.frame_set(timing[2], subframe=timing[3])
        for obj in context.selected_objects:
            obj.select_set(False)
        for obj in selected:
            obj.select_set(True)
        context.view_layer.objects.active = active
        if mode != "OBJECT":
            bpy.ops.object.mode_set(mode=mode)

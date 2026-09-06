"""Create Blender actions from the original sampled NMA skeletal tracks."""

from __future__ import annotations

import bpy
from mathutils import Euler, Matrix, Quaternion

if __package__:
    from .morphs_to_blender import create_facial_channels, assign_facial_action
    from .nmba_animation import Animation, load_animation
else:
    from morphs_to_blender import create_facial_channels, assign_facial_action
    from nmba_animation import Animation, load_animation


def node_paths(nodes):
    paths = {}

    def path(identifier):
        if identifier not in paths:
            node = nodes[identifier]
            paths[identifier] = (path(node.parent) + "|" if node.parent != -1 else "") + node.name
        return paths[identifier]

    return {path(identifier): identifier for identifier in nodes}


def map_tracks(clip: Animation, nodes, *, skipped=None):
    paths = node_paths(nodes)
    track_parents = {track.path.rsplit("|", 1)[0] for track in clip.tracks if "|" in track.path}
    result = {}
    for track in clip.tracks:
        if track.path not in paths:
            # Shared fox clips include ten finger endpoints that variant 1
            # omits. A constant leaf with an existing parent cannot influence
            # any model node. Missing branches or animated tracks still fail.
            parent = track.path.rsplit("|", 1)[0]
            constant = all(all(v == values[0] for v in values) for values in track.channels.values())
            if (parent in paths and track.path not in track_parents and constant
                    and not any(path.startswith(track.path + "|") for path in paths)):
                if skipped is not None:
                    skipped.append(track.path)
                continue
            raise ValueError(f"animation node is absent from the model: {track.path}")
        result[paths[track.path]] = track
    return result


def local_transform(node, track=None, sample=0):
    """Translation is absolute; Euler rotation is relative to joint orientation."""
    position = track.value(0x10, sample) if track else node.translation
    scale = track.value(0x0E, sample) if track else node.scale
    x, y, z, w = node.rotation
    orientation = Quaternion((w, x, y, z)).normalized().to_matrix().to_4x4()
    if track:
        delta = Euler(track.value(0x0F, sample), track.rotation_order).to_matrix().to_4x4()
    else:
        px, py, pz, pw = node.pre_rotation
        delta = Quaternion((pw, px, py, pz)).normalized().to_matrix().to_4x4()
    return Matrix.Translation(position) @ orientation @ delta @ Matrix.Diagonal((*scale, 1))


def create_action(clip, armature, nodes, bind_worlds, conversion):
    skipped = []
    tracks = map_tracks(clip, nodes, skipped=skipped)
    bones = {int(bone["nmb_node_id"]): bone for bone in armature.data.bones}
    if not any(identifier in bones for identifier in tracks):
        raise ValueError(f"{clip.name} has no tracks for this armature")
    # Mesh-branch nodes are separate from the skeleton. Do not silently drop
    # animation on them if a future asset starts using their transforms.
    for identifier, track in tracks.items():
        if identifier in bones:
            continue
        rest = local_transform(nodes[identifier])
        for sample in range(len(clip.times)):
            matrix = local_transform(nodes[identifier], track, sample)
            if max(abs(matrix[r][c] - rest[r][c]) for r in range(4) for c in range(4)) > 1e-5:
                raise ValueError(f"animated mesh-branch node is unsupported: {track.path}")

    action = bpy.data.actions.new(clip.name)
    action.use_fake_user = True
    action["nma_fps"] = clip.fps
    action["nma_source_start"] = clip.times[0]
    action["nma_extra_chunks"] = ",".join(clip.extra_chunks)
    action["nma_skipped_endpoints"] = "\n".join(skipped)
    # Slotted actions are the native API in Blender 4.4 and later.
    slot = action.slots.new(id_type="OBJECT", name=armature.name)
    layer = action.layers.new("NMA skeletal motion")
    strip = layer.strips.new(type="KEYFRAME")
    curves = strip.channelbags.new(slot).fcurves
    frames = [1 + t - clip.times[0] for t in clip.times]

    for identifier, bone in bones.items():
        node = nodes[identifier]
        track = tracks.get(identifier)
        # The rig rotates game joint X into Blender bone Y. Conjugate the
        # source animation by that basis so axes and rest transforms agree.
        axes = (conversion @ bind_worlds[identifier]).inverted() @ bone.matrix_local
        # Parent and child skin binds need not equal their authored default
        # transforms. Derive the local basis from both Blender rest matrices
        # instead of assuming bind_local == default_NODE_local.
        if bone.parent:
            parent_id = int(bone.parent["nmb_node_id"])
            parent_axes = ((conversion @ bind_worlds[parent_id]).inverted()
                           @ bone.parent.matrix_local)
            rest_local = bone.parent.matrix_local.inverted() @ bone.matrix_local
            prefix = rest_local.inverted() @ parent_axes.inverted()
        else:
            prefix = bone.matrix_local.inverted() @ conversion
        values = [[] for _ in range(10)]
        previous_rotation = None
        for sample, frame in enumerate(frames):
            basis = prefix @ local_transform(node, track, sample) @ axes
            location, rotation, scale = basis.decompose()
            if previous_rotation is not None and previous_rotation.dot(rotation) < 0:
                rotation.negate()
            previous_rotation = rotation.copy()
            for channel, value in zip(values, (*location, *rotation, *scale)):
                channel.extend((frame, value))

        pose_bone = armature.pose.bones[bone.name]
        pose_bone.rotation_mode = "QUATERNION"
        offset = 0
        for property_name, width in (("location", 3), ("rotation_quaternion", 4), ("scale", 3)):
            for component in range(width):
                curve = curves.new(pose_bone.path_from_id(property_name), index=component)
                curve.keyframe_points.add(len(frames))
                curve.keyframe_points.foreach_set("co", values[offset + component])
                # Input clips already contain one sample per game frame.
                # Avoid Bezier overshoot when playing these baked samples.
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"
                curve.update()
            offset += width
    create_facial_channels(action, clip, armature)
    action.use_frame_range = True
    action.frame_start, action.frame_end = frames[0], frames[-1]
    print(f"Animation: {clip.name}, {len(frames)} samples at {clip.fps:g} fps")
    return action


def activate_action(armature, action):
    armature.animation_data_create()
    armature.animation_data.action = action
    armature.animation_data.action_slot = action.slots[0]
    assign_facial_action(armature, action)
    scene = bpy.context.scene
    fps = float(action["nma_fps"])
    scene.render.fps = round(fps)
    scene.render.fps_base = round(fps) / fps
    scene.frame_start = round(action.frame_start)
    scene.frame_end = round(action.frame_end)
    scene.frame_set(scene.frame_start)


def load_actions(armature, nodes, bind_worlds, paths, conversion):
    clips = [load_animation(path) for path in paths]
    if len({clip.fps for clip in clips}) > 1:
        raise ValueError("clips with different frame rates must be exported separately")
    if len({clip.name for clip in clips}) != len(clips):
        raise ValueError("animation filenames must have distinct stems")
    actions = [create_action(clip, armature, nodes, bind_worlds, conversion) for clip in clips]
    if actions:
        activate_action(armature, actions[0])
    extras = sorted({chunk for clip in clips for chunk in clip.extra_chunks})
    if extras:
        print(f"Extra animation chunks not imported: {', '.join(extras)}")
    skipped = {path for action in actions for path in action["nma_skipped_endpoints"].splitlines()}
    if skipped:
        print(f"Skipped {len(skipped)} constant animation endpoints absent from this model variant")
    return actions
